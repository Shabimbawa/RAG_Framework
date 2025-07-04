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

TITLE_PROMPT_TEMPLATE = """
You are a helpful assistant. Based on the inputs below, generate a short, concise, and clear **table title** (5–15 words maximum). Avoid using redundant phrases or long explanations.

### Inputs
- Previous Span: {prev_span}
- Next Span: {next_span}
- Section Header: {section_header}

### Guidelines:
- Prefer phrasing like "Summary of...", "Breakdown of...", "Overview of...", etc.
- Avoid legal-style or overly verbose text.
- Skip unnecessary context like “read in conjunction with” or regulatory language.

### Output:
Return ONLY the concise title — no markdown, no commentary, no quotes.
"""


def extract_year_from_filename(filename):
    match = re.search(r"-(\d{4})", filename)
    return match.group(1) if match else None

def clean_column_name(col):
    col = col.strip().replace(" ", "_")
    col = re.sub(r"[:%$(),]", "", col)
    if re.match(r"^\d", col):
        col = f"year_{col}"
    return col

def parse_numeric_value(value):
    if isinstance(value, str):
        match = re.search(r"[-\d.,]+", value.replace(",", ""))
        if match:
            try:
                return float(match.group()), value
            except ValueError:
                return None, value
    return None, value

def generate_title(prev_span: str, next_span: str, section_header: str) -> str:
    prompt = ChatPromptTemplate.from_template(TITLE_PROMPT_TEMPLATE)
    chain = prompt | chat | StrOutputParser()
    title = chain.invoke({
        "prev_span": prev_span,
        "next_span": next_span,
        "section_header": section_header
    })
    return title.strip()


def insert_table_data(json_data):
    metadata = json_data["metadata"]
    data_rows = json_data["data"]

    company_ticker = metadata["company_ticker"]
    form_type = metadata["form_type"]
    year = extract_year_from_filename(metadata["source_file"])
    title = generate_title(
    metadata.get("prev_span", ""),
    metadata.get("next_span", ""),
    metadata.get("section_header", "")
)

    with driver.session() as session:
        # Match existing year node
        result = session.run("""
            MATCH (c:Company {company_ticker: $company_ticker})
            -[:HAS_FILING]->(f:FilingType {form_type: $form_type})
            -[:HAS_YEAR]->(y:Year {filing_year: $year})
            RETURN y
        """, {"company_ticker": company_ticker, "form_type": form_type, "year": year})

        year_node = result.single()
        if not year_node:
            print(f"Year node not found for {company_ticker} {form_type} {year}")
            return

        # Create Table node
        table_params = {
            "source_file": metadata["source_file"],
            "company_ticker": company_ticker,
            "form_type": form_type,
            "table_index": metadata["table_index"],
            "extraction_date": metadata["extraction_date"],
            "title": title
        }
        session.run("""
            MERGE (t:Table {
                source_file: $source_file,
                company_ticker: $company_ticker,
                form_type: $form_type,
                table_index: $table_index
            })
            SET t.extraction_date = $extraction_date,
                t.title = $title
        """, table_params)

        # Link Table to Year
        session.run("""
            MATCH (t:Table {source_file: $source_file, company_ticker: $company_ticker, form_type: $form_type, table_index: $table_index})
            MATCH (c:Company {company_ticker: $company_ticker})-[:HAS_FILING]->(:FilingType {form_type: $form_type})-[:HAS_YEAR]->(y:Year {filing_year: $year})
            MERGE (y)-[:HAS_TABLE]->(t)
        """, {**table_params, "year": year})

        # Clean column headers
        headers = [clean_column_name(h) for h in metadata["column_headers"] if not h.startswith("col_") and h.strip() != ""]

        for row in data_rows:
            props = {}
            for key, val in row.items():
                clean_key = clean_column_name(key)
                if clean_key in headers:
                    parsed_val, original_val = parse_numeric_value(val)
                    props[clean_key] = parsed_val if parsed_val is not None else val
                    props[f"{clean_key}_original"] = original_val

            session.run("""
                MATCH (t:Table {source_file: $source_file, company_ticker: $company_ticker, form_type: $form_type, table_index: $table_index})
                CREATE (d:DataRow $props)
                CREATE (t)-[:HAS_DATA]->(d)
            """, {"source_file": metadata["source_file"],
                  "company_ticker": company_ticker,
                  "form_type": form_type,
                  "table_index": metadata["table_index"],
                  "props": props})

def process_all_files():
    for filename in sorted(os.listdir(JSON_DIR)):
        if filename.endswith(".json"):
            filepath = os.path.join(JSON_DIR, filename)
            with open(filepath, 'r') as f:
                json_data = json.load(f)
                insert_table_data(json_data)
                print(f"Processed {filename}")

if __name__ == "__main__":
    process_all_files()
    driver.close()
