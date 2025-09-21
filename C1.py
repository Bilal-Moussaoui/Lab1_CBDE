# pip install sentence-transformers chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time, statistics
from tqdm import tqdm

# Cargar el BookCorpus y darle formato para Chroma
ds = load_dataset("PatrickHaller/wiki-and-book-corpus-10M", split="train[:10000]")

chunks = list(ds["train"])
chunks_ids = [str(i) for i in range(1, len(chunks) + 1)]  # Ids de los chunks: "1".."10000"

# Calcular embeddings localmente
print("[C1] Calculando embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims
embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True).tolist()

# Conectar a las colecciones creadas en C0
client = PersistentClient(path="./chroma_persist")
collections = {
    "euclidean": client.get_or_create_collection(name="lab1_chunks_euclidean"),
    "cosine": client.get_or_create_collection(name="lab1_chunks_cosine")
}

BATCH_SIZE = 1000
total = len(chunks)

# Función para actualizar embeddings en una colección y medir tiempos
def update_embeddings_and_measure(collection_name, collection):
    print(f"\n[C1] Actualizando embeddings en colección {collection_name}...")
    times = []
    
    for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Actualización {collection_name}"):
        batch_ids = chunks_ids[i:i+BATCH_SIZE]
        batch_embs = embs[i:i+BATCH_SIZE]
        
        t0 = time.perf_counter()
        collection.update(ids=batch_ids, embeddings=batch_embs)
        times.append(time.perf_counter() - t0)

    # Métricas de inserción de embeddings
    t_total = sum(times)
    t_min = min(times) if times else 0.0
    t_max = max(times) if times else 0.0
    t_avg = statistics.mean(times) if times else 0.0
    t_std = statistics.pstdev(times) if len(times) > 1 else 0.0

    print(f"\n[C1] Resultados inserción de embeddings colección {collection_name}")
    print(f"Documentos: {total}")
    print(f"Nº de lotes: {len(times)} (tamaño de los lotes: {BATCH_SIZE})")
    print(f"Tiempo total inserción: {t_total:.3f} s")
    print(f"Lote - min: {t_min:.4f} s")
    print(f"Lote - max: {t_max:.4f} s")
    print(f"Lote - std: {t_std:.4f} s")
    print(f"Lote - media: {t_avg:.4f} s")
    
    return t_total

# Actualizar embeddings en ambas colecciones
euclidean_time = update_embeddings_and_measure("euclidean", collections["euclidean"])
cosine_time = update_embeddings_and_measure("cosine", collections["cosine"])

# Resumen comparativo
print(f"\n[C1] Resumen comparativo inserción de embeddings:")
print(f"Tiempo total euclidean: {euclidean_time:.3f} s")
print(f"Tiempo total cosine: {cosine_time:.3f} s")
