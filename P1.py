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


# Cursor para interactuar con la base de datos y crear la tabla embeddings_table si aún no existe.
# En esta tabla se almacenarán los embeddings de los chunks de texto de nuestro corpus. Importante id INTEGER PRIMARY KEY REFERENCES chunks_table(id), para que se relacione con la tabla chunks_table.
cursor = postgres_connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS embeddings_table (
  id SERIAL PRIMARY KEY REFERENCES chunks_table(id),
  embedding REAL[] NOT NULL
);
""")
postgres_connection.commit()

# Cargar el corpus de texto y obtener los embeddings de cada chunk.
t0 = time.time()
cursor.execute("SELECT id, chunk FROM chunks_table ORDER BY id")
chunks = cursor.fetchall()          # [(id, chunk), ...]
t1 = time.time()
print(f"Devueltos {len(chunks)} chunks en {t1 - t0:.4f} segundos")


# Carga del modelo de embeddings de Hugging Face.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# El modelo de embeddings espera una lista de strings, no una lista de tuplas.
chunk_ids = [row[0] for row in chunks]
string_chunks = [row[1] for row in chunks]           

# model.encode() devuelve numpy.ndarray.
embeddings_np = model.encode(string_chunks, convert_to_numpy=True, show_progress_bar=True)

# Debemos convertirlo a lista de listas de floats y por cada lista de floats crear una tupla para insertar en PostgreSQL.
# [(id, emb_del_primer_chunk), (id, emb_del_segundo_chunk), ...]
embeddings_py = embeddings_np.tolist()
embeddings_tuples = [(chunk_id, embedding) for chunk_id, embedding in zip(chunk_ids, embeddings_py)]


# Insertar los embeddings en la tabla embeddings_table por lotes para obtener métricas detalladas
# On conflict do update es para actualizar el embedding si ya existe el id.
batch_size = 1000
times = []
total = len(embeddings_tuples)

for i in tqdm(range(0, total, batch_size)):
    batch_embeddings = embeddings_tuples[i:i+batch_size]
    
    t0 = time.perf_counter()
    execute_values(
        cursor,
        "INSERT INTO embeddings_table (id, embedding) VALUES %s "
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

print("\n[P1] Resultados inserción de los embeddings en PostgreSQL")
print(f"Documentos: {total}")
print(f"Nº de lotes: {len(times)} (tamaño de los lotes: {batch_size})")
print(f"Tiempo total inserción: {t_total:.3f} s")
print(f"Lote - min: {t_min:.4f} s")
print(f"Lote - max: {t_max:.4f} s")
print(f"Lote - std: {t_std:.4f} s")
print(f"Lote - media: {t_avg:.4f} s")


# Cerrar la conexión con la base de datos.
cursor.close()
postgres_connection.close()
