from datasets import load_dataset
import psycopg2, time
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer


# Primer paso: Establecer conexión con el servidor Postgres hosteado en local (localhost) mediante psycopg2:
postgres_connection = psycopg2.connect(
    host="localhost",
    database="cbde_database",
    user="postgres",
    password="postgres",
    port="5432"
)


# Segundo paso: Crear un cursor para interactuar con la base de datos y crear la tabla embeddings_table si aún no existe.
# En esta tabla se almacenarán los embeddings de los chunks de texto de nuestro corpus.
cursor = postgres_connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS embeddings_table (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  embedding REAL[] NOT NULL
);
""")
postgres_connection.commit()


# Tercer paso: Cargar el modelo de embeddings de Hugging Face.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Cuarto paso: Cargar el corpus de texto y obtener los embeddings de cada chunk.
cursor.execute("SELECT chunk FROM chunks_table") # Ejecutar la consulta y obtener los chunks
chunks = cursor.fetchall()   # lista de tuplas con los chunks

string_chunks = []           # Convertir la lista de tuplas en una lista de strings para poderla pasar al modelo de embeddings
for row in chunks:
    string_chunks.append(row[0])

embeddings_np = model.encode(string_chunks)

embeddings_py = embeddings_np.tolist()   # Convertir el numpy.ndarray en una lista de listas de floats (python) para poderla insertar en la tabla embeddings_table (PostgreSQL no entiende de numpy.ndarray)

# Cada fila debe ser una TUPLA para poderla insertar en la tabla embeddings_table.
embeddings_tuples = [(emb_list,) for emb_list in embeddings_py]


# Quinto paso: Insertar los embeddings en la tabla embeddings_table y medir el tiempo de ejecución.
start_time = time.time()
execute_values(
    cursor,
    "INSERT INTO embeddings_table (embedding) VALUES %s",
    embeddings_tuples,                 # [(emb_of_the_first_chunk,), (emb_of_the_second_chunk,), ...]
    page_size=1000
)
postgres_connection.commit()
end_time = time.time()
print(f"Insertados {len(embeddings_tuples)} embeddings en {end_time - start_time} segundos")


# Sexto paso: Cerrar la conexión con la base de datos.
cursor.close()
postgres_connection.close()
