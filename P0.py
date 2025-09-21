from datasets import load_dataset
import psycopg2, time, statistics
from psycopg2.extras import execute_values
from tqdm import tqdm

# Cargar el Bookcorpus elegido, es nuestro caso: PatrickHaller/wiki-and-book-corpus-10M de Hagging Face y seleccionar 10000 frases para trabajar.
ds = load_dataset("PatrickHaller/wiki-and-book-corpus-10M", split="train[:10000]")

# Establecer conexión con el servidor Postgres hosteado en local (localhost) mediante psycopg2:
postgres_connection = psycopg2.connect(
    host="localhost",
    database="cbde_database",
    user="postgres",
    password="postgres",
    port="5432"
)

# Crear un cursor para interactuar con la base de datos y crear la tabla chunks_table. 
# La tabla chunks_table almacenará los chunks de texto de las sentencias de nuestro corpus.
cursor = postgres_connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks_table (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  chunk TEXT NOT NULL
);
""")
postgres_connection.commit()

# Seleccionar los chunks y insertarlos en la tabla chunks_table.
chunks = [(string_chunk,) for string_chunk in ds["train"]]  # Seleccionar los strings (chunk) y convierte cada string en tupla de 1 elemento

# Medición por lotes para sacar las métricas de inserción de los chunks en PostgreSQL
batch_size = 1000
times = []
total = len(chunks)

for i in tqdm(range(0, total, batch_size)):
    batch_chunks = chunks[i:i+batch_size]
    
    t0 = time.perf_counter()
    execute_values(cursor,
        "INSERT INTO chunks_table (chunk) VALUES %s",
        batch_chunks,
        page_size=batch_size
    )
    postgres_connection.commit()
    times.append(time.perf_counter() - t0)

# Métricas de inserción
t_total = sum(times)
t_min = min(times) if times else 0.0
t_max = max(times) if times else 0.0
t_avg = statistics.mean(times) if times else 0.0
t_std = statistics.pstdev(times) if len(times) > 1 else 0.0

print("\n[P0] Resultados inserción de los chunks en PostgreSQL")
print(f"Documentos: {total}")
print(f"Nº de lotes: {len(times)} (tamaño de los lotes: {batch_size})")
print(f"Tiempo total: {t_total:.3f} s")
print(f"Lote - min: {t_min:.4f} s")
print(f"Lote - max: {t_max:.4f} s")
print(f"Lote - std: {t_std:.4f} s")
print(f"Lote - media: {t_avg:.4f} s")

# Cerrar la conexión con la base de datos.
cursor.close()
postgres_connection.close()

