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

# Textos de consulta arbitrarios
query_texts = [
    "what are you looking for ?",
    "She opened the window to let the fresh air in.",
    "Do you know the way to the station?",
    "He couldn’t believe how quiet the city was at night.",
    "Please pass me the salt.",
    "The children were playing in the garden until sunset.",
    "He looked at the stars and wondered about the universe.",
    "It started raining just as we reached the park.",
    "Can you help me carry these boxes?",
    "is this where you live ?"
]


print("[C2] Generando embeddings de queries...")
query_embeddings = model.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)

def run_query(collection, query_embedding, n_results=2):
    t0 = time.perf_counter()
    # Usar embedding directo
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    # Tomar los primeros n_results directamente
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
    
    for (query_text, query_embedding) in tqdm(zip(query_texts, query_embeddings), desc=f"Búsquedas {collection_name}"):
        # Hacer búsqueda de similitud usando embedding pre-generado
        query_results, t_query = run_query(collection, query_embedding)
        query_times.append(t_query)
        
        results.append({
            "query_text": query_text,
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

for i in range(len(query_texts)):
    print(f"\nQuery {i + 1}: {query_texts[i]}")
    
    # Resultados euclidean
    euclidean_ids = euclidean_results[i]["results"]["ids"][0]
    euclidean_distances = euclidean_results[i]["results"]["distances"][0]
    euclidean_docs = euclidean_results[i]["results"]["documents"][0]
    euclidean_chroma_results = list(zip(euclidean_ids, euclidean_distances))
    
    # Resultados cosine
    cosine_ids = cosine_results[i]["results"]["ids"][0]
    cosine_distances = cosine_results[i]["results"]["distances"][0]
    cosine_docs = cosine_results[i]["results"]["documents"][0]
    cosine_chroma_results = list(zip(cosine_ids, cosine_distances))
    
    print(f"  Euclidean: {euclidean_chroma_results} ({euclidean_results[i]['time']:.5f}s)")
    if len(euclidean_docs) >= 2:
        print("First neighbor:", euclidean_docs[0])
        print("Second neighbor:", euclidean_docs[1])
    elif len(euclidean_docs) == 1:
        print("First neighbor:", euclidean_docs[0])

    print(f"  Cosine:    {cosine_chroma_results} ({cosine_results[i]['time']:.5f}s)")
    if len(cosine_docs) >= 2:
        print("First neighbor:", cosine_docs[0])
        print("Second neighbor:", cosine_docs[1])
    elif len(cosine_docs) == 1:
        print("First neighbor:", cosine_docs[0])

# Resumen comparativo
print(f"\n[C2] Resumen comparativo búsquedas:")
print(f"Tiempo total euclidean: {euclidean_total_time:.5f} s")
print(f"Tiempo total cosine: {cosine_total_time:.5f} s")