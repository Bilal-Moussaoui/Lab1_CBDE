import time, psycopg2, statistics
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

postgres_connection = psycopg2.connect(
    host="localhost",
    database="cbde_database",
    user="postgres",
    password="postgres",
    port="5432"
)
cursor = postgres_connection.cursor()

# Inicializar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Textos de chunks para generar embeddings
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

def run_similarity_search(metric, query_embedding):
    """Ejecuta búsqueda de similitud usando pgvector"""
    t0 = time.perf_counter()
    
    if metric == "euclidean":
        # Usar distancia euclidiana (L2) con pgvector
        cursor.execute("""
            SELECT e.id, e.embedding <-> %s::vector AS distance
            FROM embeddings_table_pgvector e
            ORDER BY e.embedding <-> %s::vector
            LIMIT 2
        """, (query_embedding, query_embedding))
    elif metric == "cosine":
        # Usar distancia coseno con pgvector
        cursor.execute("""
            SELECT e.id, 1 - (e.embedding <=> %s::vector) AS distance
            FROM embeddings_table_pgvector e
            ORDER BY e.embedding <=> %s::vector
            LIMIT 2
        """, (query_embedding, query_embedding))
    
    neighbors = cursor.fetchall()  # [(id, distance), (id, distance)]
    dt = time.perf_counter() - t0
    return neighbors, dt

results = []
query_times = {"euclidean": [], "cosine": []}

for i, chunk_text in enumerate(tqdm(query_texts, desc="Procesando queries")):
    # Generar embedding del texto de consulta (no dependemos de que exista en la BD)
    query_embedding = model.encode(chunk_text).tolist()
    
    # Búsquedas de similitud
    euclidean_neighbors, t_euclidean = run_similarity_search('euclidean', query_embedding)
    cosine_neighbors, t_cosine = run_similarity_search('cosine', query_embedding)
    
    # Guardar tiempos para estadísticas
    query_times["euclidean"].append(t_euclidean)
    query_times["cosine"].append(t_cosine)
    
    results.append({
        "query": chunk_text,
        "euclidean": euclidean_neighbors,
        "t_euclidean": t_euclidean,
        "cosine": cosine_neighbors,
        "t_cosine": t_cosine
    })

# Estadísticas de rendimiento
def print_stats(times, metric_name):
    if times:
        t_total = sum(times)
        t_min = min(times)
        t_max = max(times)
        t_avg = statistics.mean(times)
        t_std = statistics.pstdev(times) if len(times) > 1 else 0.0
        
        print(f"\n[G2] Estadísticas búsquedas {metric_name}:")
        print(f"  Queries ejecutadas: {len(times)}")
        print(f"  Tiempo total: {t_total:.3f} s")
        print(f"  Query - min: {t_min:.4f} s")
        print(f"  Query - max: {t_max:.4f} s")
        print(f"  Query - media: {t_avg:.4f} s")
        print(f"  Query - std: {t_std:.4f} s")

print_stats(query_times["euclidean"], "euclidean")
print_stats(query_times["cosine"], "cosine")

# Mostrar resultados detallados (mismo formato que P2)
print("\n[G2] Resultados búsquedas de similitud en PostgreSQL con pgvector")
for i, r in enumerate(results):
    print(f"\nQuery {i + 1}: {r['query']}")
    
    print("  Euclidean:", r["euclidean"], f"({r['t_euclidean']:.5f}s)")
    cursor.execute("SELECT chunk FROM chunks_table_pgvector WHERE id = %s;", (r["euclidean"][0][0],))
    eucl_first_chunk = cursor.fetchone()[0]
    cursor.execute("SELECT chunk FROM chunks_table_pgvector WHERE id = %s;", (r["euclidean"][1][0],))
    eucl_second_chunk = cursor.fetchone()[0]
    print("    First neighbor:", eucl_first_chunk)
    print("    Second neighbor:", eucl_second_chunk)

    print("  Cosine:", r["cosine"], f"({r['t_cosine']:.5f}s)")
    cursor.execute("SELECT chunk FROM chunks_table_pgvector WHERE id = %s;", (r["cosine"][0][0],))
    cosine_first_chunk = cursor.fetchone()[0]
    cursor.execute("SELECT chunk FROM chunks_table_pgvector WHERE id = %s;", (r["cosine"][1][0],))
    cosine_second_chunk = cursor.fetchone()[0]
    print("    First neighbor:", cosine_first_chunk)
    print("    Second neighbor:", cosine_second_chunk)

cursor.close()
postgres_connection.close()
