import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# LLM setup
chat = ChatPerplexity(
    temperature=0,
    model="sonar-pro",
    pplx_api_key=PERPLEXITY_API_KEY,
)

# Neo4j setup
driver = GraphDatabase.driver(URI, auth=basic_auth(username, password))

# Prompt template
QUERY_PROMPT_TEMPLATE = """
You are a Cypher expert. Given a user query that includes:

- A company name (may not be exact),
- A form type (e.g., 10-K),
- A year,
- And a financial question (e.g., about revenue, net income, etc.),

Generate a Cypher query to retrieve relevant data using the following graph schema:

### Graph Schema:
- (Company)-[:HAS_FILING]->(FilingType)-[:HAS_YEAR]->(Year)-[:HAS_TABLE]->(Table)-[:HAS_DATA]->(DataRow)
- Node properties:
  - Company: company_name, company_ticker
  - FilingType: form_type
  - Year: filing_year
  - Table: title
  - DataRow: column names vary (e.g., Category, year_2020, etc.)

### Rules:
- Match company nodes using:  
  `toLower(c.company_name) CONTAINS ...` or `toLower(c.company_ticker) = ...`
- Match `form_type` and `filing_year` exactly — `filing_year` is stored as a **string**, so wrap it in double quotes (e.g., `"2020"`).
- Use `toLower(toString(...))` for any field that might be a string or number (like `Category`) to avoid errors.
- Match rows by checking `toLower(toString(row.Category))` or specific year-labeled properties (e.g., `row.year_2020`).
- Return `table.title`, `row`, or specific values as needed.
- Output only a Cypher query — no explanations, no formatting, no comments.

### User Query:
\"\"\"{user_query}\"\"\"

Cypher Query:
"""

def generate_cypher_from_query(user_query: str) -> str:
    prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
    chain = prompt | chat | StrOutputParser()
    cypher = chain.invoke({"user_query": user_query})
    return cypher.strip()

def run_cypher_query(cypher_query: str):
    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            records = result.data()
            if not records:
                print("No data found.")
            else:
                for record in records:
                    print(record)
        except Exception as e:
            print("Error running query:", e)
            print("Query was:", cypher_query)

if __name__ == "__main__":
    user_query = input("Ask your financial graph question (e.g., 'Show Apple’s net income in 2020 10-K'): ")
    cypher_query = generate_cypher_from_query(user_query)
    print("\nGenerated Cypher query:\n", cypher_query)
    print("\nQuery Results:\n")
    run_cypher_query(cypher_query)
    driver.close()
