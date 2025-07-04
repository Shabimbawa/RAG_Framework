import json
import os
import re
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
JSON_DIR = "tables_extracted/320193/10-K"

chat = ChatPerplexity(
    temperature=0,
    model="sonar-pro",
    pplx_api_key=PERPLEXITY_API_KEY,
)

driver = GraphDatabase.driver(URI, auth=basic_auth(username, password))

CYPHER_PROMPT_TEMPLATE = """ 
You are a Cypher expert. Convert the following JSON into a Cypher query to store it in a Neo4j graph.

Instructions:

- Create a parent Table node using properties from the metadata section.
    - Create an appropriate name for the parent table node using the information found in the prev_span, next_span and section_header fields
    - Ignore prev_span, next_span, or section_header fields if they contain unstructured or irrelevant text
- Each item in the data array becomes a DataRow node.
- Use the column_headers list to define property keys for each DataRow.
- Clean all property names and keys:
    - Replace spaces with underscores.
    - Remove or escape special characters like :, %, $, (, ), and commas.
    - If a key starts with a digit or contains characters that make it an invalid Cypher identifier, wrap it in backticks ( `like_this` ).
- Exclude columns from column_headers if:
- They are placeholders (e.g., "col_X").
- They are empty or have no non-empty values across all rows.
- Convert values like "$274,515" to numerical format where applicable, keeping the original string as original_value.
- Use MERGE only for the Table node to avoid duplication.
- Use CREATE for each DataRow node.
- Connect each DataRow to its parent Table node using a :HAS_DATA relationship.
- Ignore section_header, prev_span, or next_span if they contain unstructured or irrelevant text.
- Return only a single Cypher query — no markdown, no explanations, no formatting.
- Avoid placing WITH immediately after WHERE, and do not include multiple Cypher statements.

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
