from datasets import load_dataset
import psycopg2, time, statistics
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Conexión con el servidor Postgres hosteado en local (localhost) mediante psycopg2:
postgres_connection = psycopg2.connect(
    host="localhost",
    database="cbde_database",
    user="postgres",
    password="postgres",
    port="5432"
)

# Cursor para interactuar con la base de datos
cursor = postgres_connection.cursor()

# Cargar el corpus de texto y obtener los embeddings de cada chunk.
t0 = time.perf_counter()
cursor.execute("SELECT id, chunk FROM chunks_table_pgvector ORDER BY id")
chunks = cursor.fetchall()          # [(id, chunk), ...]
t1 = time.perf_counter()
print(f"Devueltos {len(chunks)} chunks en {t1 - t0:.4f} segundos")

# Carga del modelo de embeddings de Hugging Face.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# El modelo de embeddings espera una lista de strings, no una lista de tuplas.
chunk_ids = [row[0] for row in chunks]
string_chunks = [row[1] for row in chunks]           

# model.encode() devuelve numpy.ndarray.
embeddings_np = model.encode(string_chunks, convert_to_numpy=True, show_progress_bar=True)

# Convertir a lista de listas de floats para pgvector
embeddings_py = embeddings_np.tolist()

# Crear tuplas para actualizar los embeddings en la misma tabla
# [(embedding_list, id), (embedding_list, id), ...]
embeddings_tuples = [(embedding, chunk_id) for chunk_id, embedding in zip(chunk_ids, embeddings_py)]

# Actualizar los embeddings en la tabla chunks_table_pgvector por lotes para obtener métricas detalladas
batch_size = 1000
times = []
total = len(embeddings_tuples)

for i in tqdm(range(0, total, batch_size)):
    batch_embeddings = embeddings_tuples[i:i+batch_size]
    
    t0 = time.perf_counter()
    cursor.executemany(
        "UPDATE chunks_table_pgvector SET embedding = %s::vector WHERE id = %s",
        batch_embeddings
    )
    postgres_connection.commit()
    times.append(time.perf_counter() - t0)

# Métricas de inserción de embeddings
t_total = sum(times)
t_min = min(times) if times else 0.0
t_max = max(times) if times else 0.0
t_avg = statistics.mean(times) if times else 0.0
t_std = statistics.pstdev(times) if len(times) > 1 else 0.0

print("\n[G1] Resultados actualización de los embeddings en PostgreSQL con pgvector")
print(f"Documentos: {total}")
print(f"Nº de lotes: {len(times)} (tamaño de los lotes: {batch_size})")
print(f"Tiempo total actualización: {t_total:.3f} s")
print(f"Lote - min: {t_min:.4f} s")
print(f"Lote - max: {t_max:.4f} s")
print(f"Lote - std: {t_std:.4f} s")
print(f"Lote - media: {t_avg:.4f} s")

# # Índices HNSW
# # Crear índices HNSW después de insertar todos los embeddings
# print("\n[G1] Creando índices HNSW para optimizar búsquedas...")
# t_index_start = time.perf_counter()

# # Índice para distancia euclidiana (L2)
# cursor.execute("""
# CREATE INDEX IF NOT EXISTS chunks_table_pgvector_embedding_euclidean_idx 
# ON chunks_table_pgvector USING hnsw (embedding vector_l2_ops);
# """)
# postgres_connection.commit()
# print("[G1] Índice HNSW euclidean creado")

# # Índice para distancia coseno
# cursor.execute("""
# CREATE INDEX IF NOT EXISTS chunks_table_pgvector_embedding_cosine_idx 
# ON chunks_table_pgvector USING hnsw (embedding vector_cosine_ops);
# """)
# postgres_connection.commit()
# print("[G1] Índice HNSW cosine creado")

# t_index_end = time.perf_counter()
# print(f"[G1] Tiempo total creación de índices: {t_index_end - t_index_start:.3f} s")

# Índices IVFFlat
print("\n[G1] Creando índices IVFFlat para optimizar búsquedas...")
t_ivfflat_start = time.perf_counter()

# Índice IVFFlat para distancia euclidiana (L2)
cursor.execute("""
CREATE INDEX IF NOT EXISTS chunks_table_pgvector_embedding_euclidean_ivfflat_idx 
ON chunks_table_pgvector USING ivfflat (embedding vector_l2_ops) WITH (lists = 10);
""")
postgres_connection.commit()
print("[G1] Índice IVFFlat euclidean creado")

# Índice IVFFlat para distancia coseno
cursor.execute("""
CREATE INDEX IF NOT EXISTS chunks_table_pgvector_embedding_cosine_ivfflat_idx 
ON chunks_table_pgvector USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
""")
postgres_connection.commit()
print("[G1] Índice IVFFlat cosine creado")

t_ivfflat_end = time.perf_counter()
print(f"[G1] Tiempo total creación de índices IVFFlat: {t_ivfflat_end - t_ivfflat_start:.3f} s")

# Cerrar la conexión con la base de datos.
cursor.close()
postgres_connection.close()


# Observaciones:
# - CON LOS ÍNDICES ESTAMOS SACRIFICANDO PRECISIÓN (TENEMOS UNA APROXIMACIÓN) PARA GANAR VELOCIDAD. https://github.com/pgvector/pgvector
# - Los índices HNSW se crean después de insertar todos los embeddings.
# - Los índices HNSW se crean para ambas distancias: euclidiana (L2) y coseno.
# - Los índices HNSW se crean para ambas distancias: euclidiana (L2) y coseno.
# - Esto ayuda mucho a mejorar el rendimiento de las búsquedas vectoriales. (un 100% más rápido!)
# - las filas de los btree són demasiado grandes porque algunos chunks son muy largos.
# - Para los índices IVFFlat aconsejan un valor de lists = Rows/1000. 
#       De momento tengo: 10 > 50 > 100 > 500. (En cuanto a tiempos, lists = 100 es el mejor en velocidad pero no en precisión del resultado!!!!! (No devuelve los embeddings más cercanos!!!!))
#       500 sin duda ya es pasarse y se nota en que el rendimiento baja.