# pip install sentence-transformers chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm

# Conectar a la colección existente (debe tener embeddings de C1)
client = PersistentClient(path="./chroma_persist")
collection = client.get_collection("lab1_chunks_euclidean")

# Inicializar modelo para generar embeddings de queries
model = SentenceTransformer('all-MiniLM-L6-v2')


# Textos de chunks para hacer búsquedas (mismos que P2)
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

query_embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)

def run_query(query_embedding, query_text, n_results=2):
    t0 = time.perf_counter()
    # Usar embedding directo en lugar de texto (más rápido)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results + 1
    )
    
    # Si el primer resultado es idéntico al query, saltarlo
    if (results['documents'][0] and 
        results['documents'][0][0].strip() == query_text.strip()):
        # Tomar solo los siguientes n_results
        filtered_results = {
            'documents': [results['documents'][0][1:n_results+1]],
            'ids': [results['ids'][0][1:n_results+1]],
            'distances': [results['distances'][0][1:n_results+1]]
        }
    else:
        # Tomar los primeros n_results
        filtered_results = {
            'documents': [results['documents'][0][:n_results]],
            'ids': [results['ids'][0][:n_results]],
            'distances': [results['distances'][0][:n_results]]
        }
    
    dt = time.perf_counter() - t0
    return filtered_results, dt

results = []
for i, (chunk_text, query_embedding) in enumerate(tqdm(zip(chunk_texts, query_embeddings))):
    # Hacer búsqueda de similitud usando embedding pre-generado
    query_results, t_query = run_query(query_embedding, chunk_text)
    
    results.append({
        "query_id": i + 1,
        "query_text": chunk_text,
        "results": query_results,
        "time": t_query
    })

# Mostrar resultados
print("\n[C2] Resultados búsquedas de similitud en ChromaDB")
for r in results:
    print(f"\nQuery {r['query_id']}:")
    
    # Extraer solo IDs y distancias para mostrar como P2
    ids = r["results"]["ids"][0]
    distances = r["results"]["distances"][0]
    chroma_results = list(zip(ids, distances))
    
    print("ChromaDB: ", chroma_results, f"({r['time']:.4f}s)")


# Es natural que las distancias sean diferentes ya que usamos métodos difernetes para calcular las distancias.