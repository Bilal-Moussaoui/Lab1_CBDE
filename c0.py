import chromadb
from datasets import load_dataset

chroma_client = chromadb.PersistentClient(path = "./chroma_data")
collection = chroma_client.create_collection(name="chunks")

ds = load_dataset("PatrickHaller/wiki-and-book-corpus-10M", split="train[:100]")

chunks = []
for d in ds:
    chunks.append(d['train'])

print("DDDGAS")
dim = 384 #The model all-MiniLM-L6-v2 has 384 dimensions for each embedding
embeddings=[[0.0]*dim for _ in range(len(chunks))]

collection.add(
        ids = [str(j) for j in range(100)],
        documents = chunks,
        embeddings = embeddings
)

print(collection.peek())

