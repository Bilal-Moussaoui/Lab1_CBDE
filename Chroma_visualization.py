from chromadb import PersistentClient
import json

client = PersistentClient(path="./chroma_persist")
coll = client.get_collection("lab1_chunks")

# Obtener todos los datos
data = coll.get()

# Guardar en JSON
with open("chroma_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Datos guardados en chroma_data.json")
print(f"Documentos: {len(data.get('documents', []))}")
embeddings = data.get('embeddings', [])
if embeddings is None:
    print("Embeddings: 0 (no hay embeddings almacenados)")
else:
    print(f"Embeddings: {len(embeddings)}")