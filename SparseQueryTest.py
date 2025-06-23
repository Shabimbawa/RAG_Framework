from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# Your search query
query_text = "how many shares of common stock were issued and outstanding as of October 16, 2020"

# Perform sparse vector search using BM25
results = client.search(
    collection_name="sec_chunks_sparse",
    data=[query_text],  # BM25 function will convert text to sparse vector automatically
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
for i, result in enumerate(results[0]):
    print(f"Result {i+1}:")
    print(f"Score: {result['distance']}")
    print(f"Ticker: {result['entity']['ticker']}")
    print(f"Form Type: {result['entity']['form_type']}")
    print(f"Source File: {result['entity']['source_file']}")
    print(f"Text: {result['entity']['text'][:200]}...")
    print("-" * 50)