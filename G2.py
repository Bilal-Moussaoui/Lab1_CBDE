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

def run_similarity_search(metric, query_embedding, query_id):
    """Ejecuta búsqueda de similitud usando pgvector"""
    t0 = time.perf_counter()
    
    if metric == "euclidean":
        # Usar distancia euclidiana (L2) con pgvector
        cursor.execute("""
            SELECT id, embedding <-> %s::vector as distance 
            FROM chunks_table_pgvector 
            WHERE id != %s 
            ORDER BY embedding <-> %s::vector 
            LIMIT 2
        """, (query_embedding, query_id, query_embedding))
    elif metric == "cosine":
        # Usar distancia coseno con pgvector
        cursor.execute("""
            SELECT id, 1 - (embedding <=> %s::vector) as distance 
            FROM chunks_table_pgvector 
            WHERE id != %s 
            ORDER BY embedding <=> %s::vector 
            LIMIT 2
        """, (query_embedding, query_id, query_embedding))
    
    neighbors = cursor.fetchall()  # [(id, distance), (id, distance)]
    dt = time.perf_counter() - t0
    return neighbors, dt

results = []
query_times = {"euclidean": [], "cosine": []}

for i, chunk_text in enumerate(tqdm(chunk_texts, desc="Procesando queries")):
    # Generar embedding del texto del chunk
    cursor.execute("SELECT id FROM chunks_table_pgvector WHERE chunk = %s;", (chunk_text,))
    result = cursor.fetchone()
    
    query_id = result[0]
    query_embedding = model.encode(chunk_text).tolist()
    
    # Búsquedas de similitud
    euclidean_neighbors, t_euclidean = run_similarity_search('euclidean', query_embedding, query_id)
    cosine_neighbors, t_cosine = run_similarity_search('cosine', query_embedding, query_id)
    
    # Guardar tiempos para estadísticas
    query_times["euclidean"].append(t_euclidean)
    query_times["cosine"].append(t_cosine)
    
    results.append({
        "chunk_id": query_id,
        "chunk": chunk_text,
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
        print(f"  Queries: {len(times)}")
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
    print(f"\nQuery {i + 1}: (ID: {r['chunk_id']}) {r['chunk']}")
    
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
