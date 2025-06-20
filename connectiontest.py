from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
import numpy as np

connections.connect(host="localhost", port="19530")
print("Milvus Connected")

for name in ["dense", "sparse"]:
    if utility.has_collection(name):
        utility.drop_collection(name)

# Sample Data
documents = [
    "Apple is a technology company",
    "Rhenz is into men",
    "Ced is Chinese",
    "Bananas are yellow and sweet",
    "Tesla makes electric cars"
]

query = "Rhenz preference"

# Dense embedding 
dense_model = SentenceTransformer("all-MiniLM-L6-v2")
dense_vectors = dense_model.encode(documents)
dense_vectors = np.array(dense_vectors, dtype=np.float32)  
dense_query = dense_model.encode(query)
dense_query = np.array(dense_query, dtype=np.float32)

# Dense schema for Milvus storage
dense_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_vectors.shape[1]),
    FieldSchema(name="doc_index", dtype=DataType.INT64)
]
dense_schema = CollectionSchema(fields=dense_fields, description="Dense vector collection")
dense_collection = Collection(name="dense", schema=dense_schema)

dense_entities = [
    {"dense_vector": vec, "doc_index": i}
    for i, vec in enumerate(dense_vectors)
]
dense_collection.insert(dense_entities)
#print("Inserted dense vectors.")

dense_collection.create_index(
    field_name="dense_vector",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128}
    }
)
#print("Dense index created.")

# Sparse Embedding (BM25)
analyzer = build_default_analyzer(language="en")
bm25_ef = BM25EmbeddingFunction(analyzer)

bm25_ef.fit(documents)
doc_embeddings = bm25_ef.encode_documents(documents)
query_embedding = bm25_ef.encode_queries([query])[0]

def convert_sparse(embedding):
    """Convert scipy sparse matrix to {index: value} dict"""
    if hasattr(embedding, 'tocoo'):
        coo = embedding.tocoo()
        return dict(zip(coo.col, coo.data))
    return embedding  # Already a dict

doc_embeddings = [convert_sparse(emb) for emb in doc_embeddings]
query_embedding = convert_sparse(query_embedding)

# Sparse Schema
sparse_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="doc_index", dtype=DataType.INT64)
]
sparse_schema = CollectionSchema(fields=sparse_fields, description="Sparse vector collection")
sparse_collection = Collection(name="sparse", schema=sparse_schema)

# Sparse vectors into Milvus format
entities = [
    {"sparse_vector": emb, "doc_index": i}
    for i, emb in enumerate(doc_embeddings)
]
sparse_collection.insert(entities)
#print("Inserted sparse vectors.")

# Create Sparse Index
sparse_collection.create_index(
    field_name="sparse_vector",
    index_params={
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP"
    }
)
#print("Sparse index created.")

# Load Collections
dense_collection.load()
sparse_collection.load()

# Print Query
print("Query:", query)

# Dense Search
dense_results = dense_collection.search(
    data=[dense_query],  
    anns_field="dense_vector",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=1,
    output_fields=["doc_index"]
)
top_dense_index = dense_results[0][0].entity.get("doc_index")
print("Dense top result:", documents[top_dense_index])

# Sparse Search
sparse_results = sparse_collection.search(
    data=[query_embedding],  # Use converted query
    anns_field="sparse_vector",
    param={"metric_type": "IP", "params": {}},
    limit=1,
    output_fields=["doc_index"]
)

# Get result
top_index = sparse_results[0][0].entity.get("doc_index")
print("BM25 top result:", documents[top_index])