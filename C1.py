# pip install sentence-transformers chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# Cargar el BookCorpus y darle formato para Chroma
ds = load_dataset("PatrickHaller/wiki-and-book-corpus-10M", split="train[:10000]")

chunks = list(ds["train"])
chunks_ids = [str(i) for i in range(1, len(chunks) + 1)]  # Ids de los chunks: "1".."10000"

# Calcular embeddings localmente
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims
embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True).tolist()

# Guardarlos en la colección (actualizar)
client = PersistentClient(path="./chroma_persist")
collection = client.get_or_create_collection(name="lab1_chunks")  # sin embedding_function

# Insertar en lotes de 1000 (El máximo que permite Chroma es 5461)
BATCH_SIZE = 1000
for i in range(0, len(chunks), BATCH_SIZE):
    batch_ids = chunks_ids[i:i+BATCH_SIZE]
    batch_embs = embs[i:i+BATCH_SIZE]
    collection.update(ids=batch_ids, embeddings=batch_embs)
