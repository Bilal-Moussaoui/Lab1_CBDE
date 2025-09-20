import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path = "./chroma_data")
collection = chroma_client.get_collection(name="chunks")


query_texts = [
    "a loud knock echoed through the room making me raise my head .jake laughed .",
    "Maria clutched the old letter, her hands trembling with anticipation.",
    "Rain tapped gently against the window as he pondered the journey ahead.",
    "The library smelled of parchment and dust, each shelf a world of its own.",
    "He laughed, a deep, resonant sound that echoed through the empty hall.",
    "A sudden gust of wind scattered the papers across the floor.",
    "She hesitated at the doorway, unsure whether to enter or retreat.",
    "The fire crackled, warming the room and the hearts gathered around it.",
    "Footsteps echoed on the cobblestone street, a warning of approaching danger.",
    "As the clock struck midnight, the city fell silent, shrouded in mystery."
]



# EUCLIDEAN DISTANCE METRIC


print("CALCULATING The embeddings of the sentences to compare")

model_id = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)
query_embeddings = model.encode(query_texts)

print("COMPARING THE SENTENCES")

results = collection.query(
    query_embeddings = query_embeddings,
    n_results= 2
)

for i, (documents, distances) in enumerate(zip(results['documents'],results['distances'])):
    print(f"\nQuery {i+1}")
    print(f"Document: {documents}")
    print(f"Distance: {distances}")




# COSINUS DISTANCE METRIC


print("\n Now Using Cosinus distance metric")


chroma_client.delete_collection("chunks_cos")

collection_cos = chroma_client.create_collection(
    name="chunks_cos",
    configuration={
        "hnsw": {
            "space": "cosine",
        }
    }
)


docs = collection.get(include=["documents", "embeddings"])

collection_cos.add(
        ids = [str(j) for j in range(100)],
        documents = docs['documents'],
        embeddings = docs['embeddings']
)



print("\n COMPARING THE SENTENCES")

results = collection_cos.query(
    query_embeddings = query_embeddings,
    n_results= 2
)

for i, (documents, distances) in enumerate(zip(results['documents'],results['distances'])):
    print(f"\nQuery {i+1}")
    print(f"Document: {documents}")
    print(f"Distance: {distances}")

