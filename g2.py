import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import psycopg2, time
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

postgres_connection = psycopg2.connect(
    host="localhost",
    database="CBDE",
    user="postgres",
    password="postgres",
    port="5432"
)


# Cursor para interactuar con la base de datos y crear la tabla embeddings_table si aún no existe.
# En esta tabla se almacenarán los embeddings de los chunks de texto de nuestro corpus. Importante id INTEGER PRIMARY KEY REFERENCES chunks_table(id), para que se relacione con la tabla chunks_table.
cursor = postgres_connection.cursor()


query_texts = [
    "i knew if i kept at you , you 'd get snotty and mean . ''",
    "sarah looked firm again .",
    "Rain tapped gently against the window as he pondered the journey ahead.",
    "in her arrogance , she thought that nothing could halt her glorious bid for power .",
    "i climb out without looking at his face and we head for the doors .",
    "A sudden gust of wind scattered the papers across the floor.",
    "She hesitated at the doorway, unsure whether to enter or retreat.",
    "The fire crackled, warming the room and the hearts gathered around it.",
    "Footsteps echoed on the cobblestone street, a warning of approaching danger.",
    "As the clock struck midnight, the city fell silent, shrouded in mystery."
]

print("CALCULATING The embeddings of the sentences to compare")

model_id = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)
query_embeddings = model.encode(query_texts)


print("COMPARING THE SENTENCES Using L2 Distance Metric")

query_embeddings = query_embeddings.tolist()


i = 0
while i < len(query_embeddings):
    

    #Esta consulta devuelve el id de los vecinos, es decir, devuelve el id de la tabla embeddings las cuales son más parecidas al embedding pasado con %s (la frase a comparar)
    cursor.execute("""
        SELECT id FROM embeddings_table ORDER BY embedding <-> %s::vector LIMIT 2;

        """,
        (query_embeddings[i],)
    )
    l2_distance_result = cursor.fetchone()
    print("l2 distance")
    print(l2_distance_result)


    #Esta consulta devuelve el id de los vecinos, es decir, devuelve el id de la tabla embeddings las cuales son más parecidas al embedding pasado con %s (la frase a comparar)
    cursor.execute("""
        SELECT id FROM embeddings_table ORDER BY embedding <=> %s::vector LIMIT 2;

        """,
        (query_embeddings[i],)
    )
    cos_distance_result = cursor.fetchone()
    print("Cosinus")
    print(cos_distance_result)

    i+=1








