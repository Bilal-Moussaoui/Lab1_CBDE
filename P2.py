import time, psycopg2
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
# NOTA: Los IDs se obtienen dinámicamente desde la base de datos (línea 44)
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

def run_top2(metric_func, query_embedding, query_id):
    t0 = time.perf_counter()
    cursor.execute(f"SELECT neighbor_id, distance FROM {metric_func}(%s, %s);", (query_embedding, query_id))
    neighbors = cursor.fetchall()  # [(neighbor_id, distance), (neighbor_id, distance)]
    dt = time.perf_counter() - t0
    return neighbors, dt

results = []

for i, chunk_text in enumerate(tqdm(chunk_texts)):
    # Generar embedding del texto del chunk
    cursor.execute("SELECT id FROM chunks_table WHERE chunk = %s;", (chunk_text,))
    query_id = str(cursor.fetchone()[0])
    query_embedding = model.encode(chunk_text).tolist()
    
    euclidean_neighbors, t_euclidean = run_top2('top2_euclidean', query_embedding, query_id)
    manhattan_neighbors, t_manhattan = run_top2('top2_manhattan', query_embedding, query_id)

    results.append({
        "chunk_id": query_id,
        "chunk": chunk_text,
        "euclidean": euclidean_neighbors,  "t_euclidean": t_euclidean,
        "manhattan": manhattan_neighbors,  "t_manhattan": t_manhattan
    })

# Primera forma de mostrar los resultados
print("\n[P2] Resultados búsquedas de similitud en PostgreSQL")
for i,r in enumerate(results):
    print(f"\nQuery {i + 1}: (ID: {r['chunk_id']}) {r['chunk']}")
    
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

# He fet una prova molt tonta per veure que funciona bé l'algorisme:
# No excloure l'id del chunk que es rep com a paràmetre a la funció run_top2 y veure que sempre es la primera distnància, mentre que la segona es la que correspon
# a la primera distancia del resultat de l'algorisme quan sí que s'exclou. 
