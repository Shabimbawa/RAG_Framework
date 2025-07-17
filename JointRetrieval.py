from QueryRouting import classify_query
from GraphQueryTest import execute_graph_retrieval_process, close_drivers


def handle_tabular(query):
    print("Routing to Neo4j")
    execute_graph_retrieval_process(query)
    close_drivers()

def handle_textual(query):
    print("Routing to Milvus")
    # Insert function for milvus embedding retrieval

def handle_both(query): # This function still iffy, cuz im not sure how the retrieval process will work
    handle_tabular(query)
    handle_textual(query)

classification_handlers = {
    "TEXTUAL": handle_textual,
    "TABLE": handle_tabular,
    "BOTH": handle_both,
}

def route_based_on_classification(classification: str, query: str):
    handler = classification_handlers.get(classification.upper())
    handler(query)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    classification = classify_query(user_query)
    route_based_on_classification(classification, user_query)