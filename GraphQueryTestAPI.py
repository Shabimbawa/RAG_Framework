import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

Graph Schema:
- (Company)-[:HAS_FILING]->(FilingType)-[:HAS_YEAR]->(Year)-[:HAS_TABLE]->(Table)-[:HAS_DATA]->(DataRow)

Node properties:
- Company: company_name, company_ticker
- FilingType: form_type
- Year: filing_year
- Table: title
- DataRow: column names vary (e.g., Category, year_2020, etc.)

Rules:
- Match company nodes using:
  `toLower(c.company_name) CONTAINS ...` or `toLower(c.company_ticker) = ...`
- If the user query **does not explicitly mention** the form type or year:
  - Do not include those conditions in the query
  - Do not assume or guess a default form_type like "10-K"
  - Use OPTIONAL MATCH if needed, or sort years using `ORDER BY y.filing_year DESC LIMIT 1`
- When form_type or filing_year are known, match them exactly. Wrap filing_year in double quotes (e.g., "2020").
- Use `toLower(toString(...))` for fields like Category or table values to ensure compatibility.
- Return table.title, row, or specific fields as needed.
- Output the Cypher query only — do NOT include markdown formatting, backticks, or the word "cypher".

User Query:
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
            return records
        except Exception as e:
            raise RuntimeError(f"Error runnning query: {e}")
        

#API setup stuff
app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    cypher_query: str
    results: list

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    cypher_query = generate_cypher_from_query(request.user_query)
    try:
        results = run_cypher_query(cypher_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(cypher_query=cypher_query, results=results)