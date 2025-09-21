# pip install sentence-transformers chromadb
from ast import increment_lineno
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import time, statistics
from tqdm import tqdm

# Conectar a las colecciones existentes (deben tener embeddings de C1)
client = PersistentClient(path="./chroma_persist")
collections = {
    "euclidean": client.get_or_create_collection("lab1_chunks_euclidean"),
    "cosine": client.get_or_create_collection("lab1_chunks_cosine")
}

# Inicializar modelo para generar embeddings de queries
model = SentenceTransformer('all-MiniLM-L6-v2')

# Textos de chunks para hacer búsquedas (mismos que P2)
chunk_texts = [
    "i tell them .",
    "his skin was the smooth creamy tan of immortals who go into the sun often in order to pass for human , and it made his eyes appear wondrously bright and beautiful .",
    "only her .",
    "`` what are you looking for ? ''",
    "there were a few travelers eating their supper , but most were locals sitting for a drink .",
    "you have five minutes . ''",
    "is this where you live ?",
    "he stared at dog .",
    "the three strands of barbed wire on top of the six-foot-tall , chain link fence around the plant even pointed inward .",
    "i was on my own after that . '"
]

print("[C2] Generando embeddings de queries...")
query_embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)

def run_query(collection, query_embedding, query_text, n_results=2):
    t0 = time.perf_counter()
    # Usar embedding directo en lugar de texto (más rápido)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results + 1
    )
    
    # Si el primer resultado es idéntico al query, saltarlo
    if (results['documents'][0] and 
        results['documents'][0][0].strip() == query_text.strip()):
        # Tomar solo los siguientes n_results
        filtered_results = {
            'documents': [results['documents'][0][1:n_results+1]],
            'ids': [results['ids'][0][1:n_results+1]],
            'distances': [results['distances'][0][1:n_results+1]]
        }
    else:
        # Tomar los primeros n_results
        filtered_results = {
            'documents': [results['documents'][0][:n_results]],
            'ids': [results['ids'][0][:n_results]],
            'distances': [results['distances'][0][:n_results]]
        }
    
    dt = time.perf_counter() - t0
    return filtered_results, dt

def run_queries_on_collection(collection_name, collection):
    print(f"\n[C2] Ejecutando búsquedas en colección {collection_name}...")
    results = []
    query_times = []
    
    for (chunk_text, query_embedding) in tqdm(zip(chunk_texts, query_embeddings), desc=f"Búsquedas {collection_name}"):
        # Hacer búsqueda de similitud usando embedding pre-generado
        query_results, t_query = run_query(collection, query_embedding, chunk_text)
        query_times.append(t_query)
        
        results.append({
            "query_text": chunk_text,
            "results": query_results,
            "time": t_query
        })
    
    # Estadísticas de tiempos de búsqueda
    t_total = sum(query_times)
    t_min = min(query_times) if query_times else 0.0
    t_max = max(query_times) if query_times else 0.0
    t_avg = statistics.mean(query_times) if query_times else 0.0
    t_std = statistics.pstdev(query_times) if len(query_times) > 1 else 0.0
    
    print(f"\n[C2] Estadísticas de búsqueda colección {collection_name}")
    print(f"Queries ejecutadas: {len(query_times)}")
    print(f"Tiempo total: {t_total:.3f} s")
    print(f"Query - min: {t_min:.4f} s")
    print(f"Query - max: {t_max:.4f} s")
    print(f"Query - std: {t_std:.4f} s")
    print(f"Query - media: {t_avg:.4f} s")
    
    return results, t_total

# Ejecutar búsquedas en ambas colecciones
euclidean_results, euclidean_total_time = run_queries_on_collection("euclidean", collections["euclidean"])
cosine_results, cosine_total_time = run_queries_on_collection("cosine", collections["cosine"])

# Mostrar resultados detallados para ambas colecciones
print(f"\n[C2] Resultados detallados búsquedas de similitud")

for i in range(len(chunk_texts)):
    
    # Query Chroma to retrieve the id of the chunk
    result = collections["cosine"].query(query_texts=[chunk_texts[i]], include=[], n_results=1)
    chunk_id = result['ids'][0] if result['ids'] else "No encontrado"
    print(f"\nQuery {i + 1}: (ID: {chunk_id}) {chunk_texts[i]}")
    
    # Resultados euclidean
    euclidean_ids = euclidean_results[i]["results"]["ids"][0]
    euclidean_distances = euclidean_results[i]["results"]["distances"][0]
    euclidean_chroma_results = list(zip(euclidean_ids, euclidean_distances))
    
    # Resultados cosine
    cosine_ids = cosine_results[i]["results"]["ids"][0]
    cosine_distances = cosine_results[i]["results"]["distances"][0]
    cosine_chroma_results = list(zip(cosine_ids, cosine_distances))
    
    print(f"  Euclidean: {euclidean_chroma_results} ({euclidean_results[i]['time']:.5f}s)")
    print(f"  Cosine:    {cosine_chroma_results} ({cosine_results[i]['time']:.5f}s)")

# Resumen comparativo
print(f"\n[C2] Resumen comparativo búsquedas:")
print(f"Tiempo total euclidean: {euclidean_total_time:.5f} s")
print(f"Tiempo total cosine: {cosine_total_time:.5f} s")

# Es natural que las distancias sean diferentes ya que usamos métodos diferentes para calcular las distancias.