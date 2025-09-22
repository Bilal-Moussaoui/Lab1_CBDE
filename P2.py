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

def run_top2(metric_func, query_embedding):
    t0 = time.perf_counter()
    cursor.execute(f"SELECT neighbor_id, distance FROM {metric_func}(%s);", (query_embedding,))
    neighbors = cursor.fetchall()  # [(neighbor_id, distance), (neighbor_id, distance)]
    dt = time.perf_counter() - t0
    return neighbors, dt


# Estadísticas agregadas de tiempos
def print_stats(times, name):
    if not times:
        return
    t_total = sum(times)
    t_min = min(times)
    t_max = max(times)
    t_avg = statistics.mean(times)
    t_std = statistics.pstdev(times) if len(times) > 1 else 0.0
    print(f"\n[P2] Estadísticas {name}:")
    print(f"Queries ejecutadas: {len(times)}")
    print(f"Tiempo total: {t_total:.3f} s")
    print(f"Query - min: {t_min:.4f} s")
    print(f"Query - max: {t_max:.4f} s")
    print(f"Query - media: {t_avg:.4f} s")
    print(f"Query - std: {t_std:.4f} s")


results = []
euclidean_times = []
manhattan_times = []

for i, query_text in enumerate(tqdm(query_texts)):
    # Generar embedding del texto de consulta (no requiere existir en la BD)
    query_embedding = model.encode(query_text).tolist()

    euclidean_neighbors, t_euclidean = run_top2('top2_euclidean', query_embedding)
    manhattan_neighbors, t_manhattan = run_top2('top2_manhattan', query_embedding)

    results.append({
        "query": query_text,
        "euclidean": euclidean_neighbors,  "t_euclidean": t_euclidean,
        "manhattan": manhattan_neighbors,  "t_manhattan": t_manhattan
    })
    euclidean_times.append(t_euclidean)
    manhattan_times.append(t_manhattan)

print_stats(euclidean_times, "Euclidean")
print_stats(manhattan_times, "Manhattan")

print("\n[P2] Resultados búsquedas de similitud en PostgreSQL")
for i,r in enumerate(results):
    print(f"\nQuery {i + 1}: {r['query']}")
    
    print("  Euclidean:", r["euclidean"], f"({r['t_euclidean']:.5f}s)")
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["euclidean"][0][0],))
    eucl_first_chunk = cursor.fetchone()[0]
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["euclidean"][1][0],))
    eucl_second_chunk = cursor.fetchone()[0]
    print("    First neighbor:", eucl_first_chunk)
    print("    Second neighbor:", eucl_second_chunk)

    print("  Manhattan:", r["manhattan"], f"({r['t_manhattan']:.5f}s)")
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["manhattan"][0][0],))
    manh_first_chunk = cursor.fetchone()[0]
    cursor.execute("SELECT chunk FROM chunks_table WHERE id = %s;", (r["manhattan"][1][0],))
    manh_second_chunk = cursor.fetchone()[0]
    print("    First neighbor:", manh_first_chunk)
    print("    Second neighbor:", manh_second_chunk)

cursor.close()
postgres_connection.close()
