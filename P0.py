from datasets import load_dataset
import psycopg2, time
from psycopg2.extras import execute_values

# Primer paso: Cargar el Bookcorpus elegido, es nuestro caso: PatrickHaller/wiki-and-book-corpus-10M de Hagging Face y seleccionar 10000 frases para trabajar.
ds = load_dataset("PatrickHaller/wiki-and-book-corpus-10M")

# Segundo paso: Establecer conexión con el servidor Postgres hosteado en local (localhost) mediante psycopg2:
postgres_connection = psycopg2.connect(
    host="localhost",
    database="cbde_database",
    user="postgres",
    password="postgres",
    port="5432"
)

# Tercer paso: Crear un cursor para interactuar con la base de datos y crear la tabla chunks_table. 
# La tabla chunks_table almacenará los chunks de texto de las sentencias de nuestro corpus.
cursor = postgres_connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chunks_table (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  chunk TEXT NOT NULL
);
""")
postgres_connection.commit()

# Cuarto paso: Seleccionar los chunks y insertarlos en la tabla chunks_table.
chunk_dictionary = ds["train"][:10000]   # Seleccionamos 10000 frases para trabajar
chunks = [(string_chunk,) for string_chunk in chunk_dictionary["train"]]  # Seleccionar los strings (chunk) y convierte cada string en tupla de 1 elemento

t0 = time.perf_counter()
execute_values(cursor,
    "INSERT INTO chunks_table (chunk) VALUES %s",
    chunks,
    page_size=1000   # inserta en lotes de 1000 filas (Es un parámetro de tunning, se busca balancear, ni 10000 ni 1 ya que una inserción por chunk es demasiado lento)
)
postgres_connection.commit()
t1 = time.perf_counter()

print(f"Insertadas {len(chunks)} filas en {t1 - t0:.3f} segundos")

# Quinto paso: Cerrar la conexión con la base de datos.
cursor.close()
postgres_connection.close()

