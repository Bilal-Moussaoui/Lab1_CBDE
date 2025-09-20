import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


chroma_client = chromadb.PersistentClient(path = "./chroma_data")
collection = chroma_client.get_collection(name="chunks")

docs = collection.get(include=["documents"])

print("CALCULATING")
model_id = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)
embeddings = model.encode(docs['documents'])

collection.update(
        ids = [str(j) for j in range(100)],
        embeddings = embeddings
)




