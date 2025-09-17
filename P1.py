from datasets import load_dataset
import psycopg2, time
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

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
  id INTEGER PRIMARY KEY REFERENCES chunks_table(id),
  embedding REAL[] NOT NULL
);
""")
postgres_connection.commit()

# Cargar el corpus de texto y obtener los embeddings de cada chunk.
cursor.execute("SELECT id, chunk FROM chunks_table ORDER BY id")
chunks = cursor.fetchall()          # [(id, chunk), ...]


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


# Insertar los embeddings en la tabla embeddings_table y medir el tiempo de ejecución.
# On conflict do update es para actualizar el embedding si ya existe el id.
start_time = time.time()
execute_values(
    cursor,
    "INSERT INTO embeddings_table (id, embedding) VALUES %s "
    "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
    embeddings_tuples,
    page_size=1000
)
postgres_connection.commit()
end_time = time.time()
print(f"Insertados {len(embeddings_tuples)} embeddings en {end_time - start_time} segundos")


# Cerrar la conexión con la base de datos.
cursor.close()
postgres_connection.close()

# He fet una prova molt tonta per veure que funciona bé l'algorisme:
# No excloure l'id del chunk que es rep com a paràmetre a la funció run_top2 y veure que sempre es la primera distnància, mentre que la segona es la que correspon
# a la primera distancia del resultat de l'algorisme quan sí que s'exclou. 