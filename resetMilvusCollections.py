from pymilvus import MilvusClient
import os
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
print(client.list_collections())

collection_name = "dense_miniLM"

# if collection_name in client.list_collections():
#     client.drop_collection(collection_name)
#     print(f"✅ Collection '{collection_name}' dropped.")
# else:
#     print(f"⚠️ Collection '{collection_name}' does not exist.")
    