from tqdm import tqdm
from datasets import load_dataset
from chromadb import PersistentClient
import time, statistics

# Script de Chroma (C0)
# Debo crear una conexión a la BD de Chroma
# Luego cargar el mismo chunk de datos cargados en Postgres

# Cargar el BookCorpus y darle formato para Chroma
ds = load_dataset("PatrickHaller/wiki-and-book-corpus-10M", split="train[:10000]")

chunks = list(ds["train"])
chunks_ids = [str(i) for i in range(1, len(chunks) + 1)]  # Ids de los chunks: "1".."10000"

# Chroma sin embedding_function (para que NO cree embeddings automáticamente y así medirlos en C1)
# PersistentClient es para poder reutilizar C0 en C1 y C2
client = PersistentClient(path="./chroma_persist")
collection = client.get_or_create_collection(name="lab1_chunks_euclidean", metadata={"hnsw:space": "l2"})  # sin embedding_function

# 3) Insertar (solo texto) — C0
batch = 1000
times = []
total = len(chunks)

for i in tqdm(range(0, total, batch)):
    batch_ids = chunks_ids[i:i+batch]
    batch_docs = chunks[i:i+batch]

    t0 = time.perf_counter()
    # Generar 1000 vectores de 384 dimensiones todos a 0
    batch_embeddings = [[0.0] * 384 for _ in range(len(batch_docs))]
    collection.upsert(ids=batch_ids, documents=batch_docs, embeddings=batch_embeddings)
    times.append(time.perf_counter() - t0)

# Métricas de inserción
t_total = sum(times)
t_min = min(times) if times else 0.0
t_max = max(times) if times else 0.0
t_avg = statistics.mean(times) if times else 0.0
t_std = statistics.pstdev(times) if len(times) > 1 else 0.0

print("\n[C0] Resultados inserción en Chroma")
print(f"Documentos: {total}")
print(f"Nº de lotes: {len(times)} (tamaño de los lotes: {batch})")
print(f"Tiempo total: {t_total:.3f} s")
print(f"Lote - min: {t_min:.4f} s")
print(f"Lote - max: {t_max:.4f} s")
print(f"Lote - std: {t_std:.4f} s")
print(f"Lote - media: {t_avg:.4f} s")