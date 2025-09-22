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

"""
Estructura de tablas objetivo (similar a P0/P1 pero con pgvector):
  - chunks_table_pgvector(id SERIAL PK, chunk TEXT)
  - embeddings_table_pgvector(id INT PK FK->chunks_table_pgvector(id), embedding vector(384))
"""

# Asegurar extensión pgvector
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
postgres_connection.commit()

# Crear tabla de embeddings con pgvector y FK a chunks
cursor.execute("""
CREATE TABLE IF NOT EXISTS embeddings_table_pgvector (
  id SERIAL PRIMARY KEY REFERENCES chunks_table_pgvector(id),
  embedding vector(384) NOT NULL
);
""")
postgres_connection.commit()

# Cargar chunks existentes para generar embeddings
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

embeddings_tuples = [(chunk_id, embedding) for chunk_id, embedding in zip(chunk_ids, embeddings_py)]

# Insertar/actualizar embeddings en la tabla embeddings_table_pgvector por lotes
batch_size = 1000
times = []
total = len(embeddings_tuples)

for i in tqdm(range(0, total, batch_size)):
    batch_embeddings = embeddings_tuples[i:i+batch_size]
    
    t0 = time.perf_counter()
    execute_values(
        cursor,
        "INSERT INTO embeddings_table_pgvector (id, embedding) VALUES %s "
        "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
        batch_embeddings,
        page_size=batch_size
    )
    postgres_connection.commit()
    times.append(time.perf_counter() - t0)

# Métricas de inserción de embeddings
t_total = sum(times)
t_min = min(times) if times else 0.0
t_max = max(times) if times else 0.0
t_avg = statistics.mean(times) if times else 0.0
t_std = statistics.pstdev(times) if len(times) > 1 else 0.0

print("\n[G1] Resultados inserción de embeddings en PostgreSQL con pgvector")
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
# CREATE INDEX IF NOT EXISTS embeddings_table_pgvector_embedding_euclidean_idx 
# ON embeddings_table_pgvector USING hnsw (embedding vector_l2_ops);
# """)
# postgres_connection.commit()
# print("[G1] Índice HNSW euclidean creado")

# # Índice para distancia coseno
# cursor.execute("""
# CREATE INDEX IF NOT EXISTS embeddings_table_pgvector_embedding_cosine_idx 
# ON embeddings_table_pgvector USING hnsw (embedding vector_cosine_ops);
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
CREATE INDEX IF NOT EXISTS embeddings_table_pgvector_embedding_euclidean_ivfflat_idx 
ON embeddings_table_pgvector USING ivfflat (embedding vector_l2_ops) WITH (lists = 10);
""")
postgres_connection.commit()
print("[G1] Índice IVFFlat euclidean creado")

# Índice IVFFlat para distancia coseno
cursor.execute("""
CREATE INDEX IF NOT EXISTS embeddings_table_pgvector_embedding_cosine_ivfflat_idx 
ON embeddings_table_pgvector USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
""")
postgres_connection.commit()
print("[G1] Índice IVFFlat cosine creado")

t_ivfflat_end = time.perf_counter()
print(f"[G1] Tiempo total creación de índices IVFFlat: {t_ivfflat_end - t_ivfflat_start:.3f} s")

# Cerrar la conexión con la base de datos.
cursor.close()
postgres_connection.close()