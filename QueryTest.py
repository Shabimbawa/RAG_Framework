from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from queryExpansion import expand_query, sparse_formatting, dense_formatting

reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")  # or large
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
reranker_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", ports="19530")

# SPARSE SEARCH
def sparse_retrieval_test(sparse_query:str):
    sparse_results = client.search(
        collection_name="sec_chunks_sparse",
        data=[sparse_query],  # BM25 function will convert text to sparse vector automatically
        anns_field="sparse",  # Search on the sparse vector field
        limit=5,  # Number of results to return
        output_fields=["text", "ticker", "form_type", "source_file", "chunk_id"],
        search_params={
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE"
            }
        }
    )

    # Display results
    print("Sparse Results: \n\n")
    for i, result in enumerate(sparse_results[0]):
        print(f"Result {i+1}:")
        print(f"Score: {result['distance']}")
        print(f"Ticker: {result['entity']['ticker']}")
        print(f"Form Type: {result['entity']['form_type']}")
        print(f"Source File: {result['entity']['source_file']}")
        print(f"Text: {result['entity']['text'][:200]}...")
        print("-" * 50)

    return sparse_results

#DENSE SEARCH
def dense_retrieval_test(dense_query:str):

    collection = Collection(COLLECTION_NAME)
    collection.load()
    print("Collection Loaded")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    query_embedding = model.encode(dense_query, convert_to_numpy=True)
    print("Query Embedded")

    search_params = {
        "metric_type": "L2", 
        "params": {"ef": 32}
    }

    results = collection.search(
        data = [query_embedding.tolist()],
        anns_field="dense_vector",
        param=search_params,
        limit=5,
        output_fields=["source_file", "ticker", "form_type", "chunk_id", "start_index", "end_index"]
    )

    if not results:
        print("No results found, exiting...")
        exit()
    
    
    print(f"\n Dense Results: \n' \n")
    print("-" * 80)
    csv_data = []

    for i, hit in enumerate(results[0]):
        entity = hit.entity
        text_chunk = fetch_text({
            "form_type": entity.get("form_type", ""),
            "ticker": entity.get("ticker", ""),
            "source_file": entity.get("source_file", ""),
            "chunk_id": entity.get("chunk_id", "")
        })
        snippet = text_chunk[:200]
        print(f"Result {i+1}:")
        print(f"Score: {hit['distance']}")
        print(f"Ticker: {hit['entity']['ticker']}")
        print(f"Form Type: {hit['entity']['form_type']}")
        print(f"Source File: {hit['entity']['source_file']}")
        print(f"Text: {snippet}...")
        print("-" * 50)
    
    return results

COLLECTION_NAME = "dense_miniLM"
def fetch_text(metadata):
    try:
        form_type = metadata.get("form_type", "")
        ticker = metadata.get("ticker", "")
        source_file = metadata.get("source_file", "")

        base_dir = "chunked_dense"

        form_dir = f"{form_type}-texts" if form_type else ""

        chunk_dir = os.path.join(base_dir, form_dir, ticker.lower())
        chunk_file  = os.path.join(chunk_dir, source_file)

        with open(chunk_file, 'r', encoding="utf-8") as f:
            chunks = json.load(f)
        
        chunk_id = metadata.get("chunk_id", "")
        for chunk in chunks:
            if str(chunk.get("chunk_id", "")) == str(chunk_id):
                return chunk.get("content", "")
        
        return "Chunk not found"
    except FileNotFoundError:
        return f"File not found: {chunk_file}"
    except Exception as e:
        return f"Error: {str(e)}"

def rerank(query, passages, top_k=5):
    inputs = reranker_tokenizer(
        [query] * len(passages),
        passages,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(reranker_model.device)

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)
        scores = F.softmax(scores, dim=0)

    ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def execute_vector_retrieval(user_query:str):
    psuedo_doc= expand_query(user_query)
    sparse_query= sparse_formatting(user_query, psuedo_doc)
    dense_query= dense_formatting(user_query, psuedo_doc)

    sparse_results= sparse_retrieval_test(sparse_query)
    dense_results= dense_retrieval_test(dense_query)

    compiled_results = []
    
    for result in sparse_results[0]:
        compiled_results.append(result['entity']['text'])

    for hit in dense_results[0]:
        text = fetch_text({...})  # Already in your code
        compiled_results.append(text)

    reranked = rerank(user_query, compiled_results, top_k=5)

    print("\nFinal Reranked Results:")
    for i, (text, score) in enumerate(reranked):
        print(f"Rank {i+1} - Score: {score:.4f}")
        print(text[:300], "...\n", "-"*60)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    execute_vector_retrieval(user_query)



