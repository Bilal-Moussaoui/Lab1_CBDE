from chromadb import PersistentClient
import pprint, json

client = PersistentClient(path="./chroma_persist")
coll = client.get_collection("lab1_chunks")
data = coll.get() # Gets all the data

with open("chroma_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(coll.peek())