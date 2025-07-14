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

- Extract the year from `metadata.source_file` by getting the 4 digits immediately after the first dash ("-").
- Match the correct Year node using:
    - `Company` node with `company_ticker = metadata.company_ticker`
    - `FilingType` node with `filing_type = metadata.form_type`
    - `Year` node with `year = extracted year`
- Connect the new Table node to this Year node with a `:HAS_TABLE` relationship.

### Table Node:
- Create a `Table` node with these properties:
    - `source_file`, `company_ticker`, `form_type`, `table_index`, and `extraction_date` from `metadata`.
    - A `title` derived from the `prev_span` and/or `next_span` fields:
        - If both are structured, concatenate them.
        - If only one is usable, use that.
        - Ignore if they’re irrelevant/unstructured.
    - Use `MERGE` for the `Table` node.

### DataRow Nodes:
- Each object in the `data` array becomes a new `DataRow` node.
- Use the `column_headers` list from metadata to define property keys.
- Clean all column header strings:
    - Replace spaces with underscores.
    - Remove or escape special characters like :, %, $, (, ), and commas.
    - If a key starts with a digit (e.g., "2020"), prefix it with `year_` (e.g., `year_2020`) instead of wrapping in backticks.

- Exclude columns like `"col_5"`, `"col_6"` if they are placeholders or empty across all rows.
- Parse values like `"$94.68"` or `"$56,353"` into floats (e.g. `94.68` and `56353`), but also keep the original string under an `original_value` field.

### Relationships:
- Use `MERGE` only for the Table node (to avoid duplicates).
- Use `CREATE` for each DataRow node.
- Connect each DataRow to the Table using a `:HAS_DATA` relationship.
- Connect the Table node to the matched Year node using `:HAS_TABLE`.

### Output:
- Output a single valid Cypher query — no markdown, no comments, no explanations, no formatting hints.

JSON:
{json_data}

Cypher Query:
"""

def process_json_files():
    for filename in sorted(os.listdir(JSON_DIR))[4:7]:
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
