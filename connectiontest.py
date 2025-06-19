from pymilvus import connections

connections.connect(host="localhost", port="19530")
print("Milvus Connected")