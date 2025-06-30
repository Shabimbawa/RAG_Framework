import json
import os
import re
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase, basic_auth

URI = "bolt://localhost:7687"
username = "neo4j"
password = "password123"
PERPLEXITY_API_KEY = "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"
JSON_DIR = "tables_extracted/320193/10-K"

chat = ChatPerplexity(
    temperature=0,
    model="sonar",
    pplx_api_key=PERPLEXITY_API_KEY,
)

driver = GraphDatabase.driver(URI, auth=basic_auth(username, password))

CYPHER_PROMPT_TEMPLATE = """ 
You are a Cypher expert. Convert the following JSON into a Cypher query to store it in a Neo4j graph.

Instructions:
- Create one parent node called `Table` with properties from the `metadata` section.
- Each entry in the `data` array should become a `DataRow` node.
- Use the `column_headers` list to define property names in each `DataRow`.
- Clean all property names: remove colons (`:`), %, $, and special characters. Use underscores for spacing.
- Use `MERGE` only for the `Table` node. Use `CREATE` for all `DataRow` nodes.
- Connect each `DataRow` to its `Table` using a `:HAS_DATA` relationship.
- Ignore `section_header`, `prev_span`, or `next_span` if they contain irrelevant or noisy text.
- Omit columns that are clearly placeholders or empty (like `col_1`, `col_2`) unless they contain real values.
- Return only ONE Cypher query — no explanations, no code blocks, and no formatting.
- Avoid syntax issues: do not place `WITH` immediately after `WHERE`. Do not include multiple Cypher statements.

JSON:
{json_data}

Cypher Query:
"""

def process_json_files():
    for filename in os.listdir(JSON_DIR)[:5]:
        if filename.endswith(".json"):
            filepath = os.path.join(JSON_DIR, filename)
            with open(filepath, 'r') as f:
                json_data = json.load(f)

                prompt = ChatPromptTemplate.from_template(CYPHER_PROMPT_TEMPLATE)
                chain = prompt | chat | StrOutputParser()
                cypher_query = chain.invoke({"json_data": json.dumps(json_data, indent=2)})

                clean_query = extract_cypher(cypher_query)
                execute_cypher(clean_query, json_data)
                print(f"Processed {filename}")

def extract_cypher(response):
    try:
        match = re.search(r"```(?:cypher)?\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()  # Fallback: return the whole thing
    except Exception as e:
        print(f"Error extracting Cypher: {str(e)}")
        return response

def execute_cypher(query, params=None):
    with driver.session() as session:
        session.run(query, parameters=params if params else {})
    print(f"Executed Cypher: {query[:100]}...")

if __name__ == "__main__":
    process_json_files()
    driver.close()


# def query_graph(cypher_query, parameters=None):
#     with driver.session() as session:
#         result = session.run(cypher_query, parameters or {})
#         records = result.data()
#     print(f"Query returned {len(records)} records.")
#     return records
