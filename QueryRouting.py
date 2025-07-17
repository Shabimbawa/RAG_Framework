import os
from dotenv import load_dotenv
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

# Prompt template
QUERY_PROMPT_TEMPLATE = """
You are a classification assistant designed to help decide the best source for answering a financial question. You have access to two types of data:

1. **TEXTUAL** — This includes narrative sections from SEC filings (10-K, 10-Q, 8-K), such as risk factors, business overviews, management discussions, etc.
2. **TABLE** — This includes structured financial tables (e.g., income statements, balance sheets, segment revenues) extracted and stored in a graph database.
3. **BOTH** — The question likely requires information from both narrative text and financial tables.

Your task is to classify the following **user query** into one of these categories: `TEXTUAL`, `TABLE`, or `BOTH`.

### Classification Examples:

**Example 1:**
Q: "What risks did Apple mention in its 2020 10-K filing?"
A: TEXTUAL

**Example 2:**
Q: "Show me the net sales of iPhones from 2018 to 2020."
A: TABLE

**Example 3:**
Q: "How did Apple's revenue growth correlate with its discussion of market uncertainty in 2020?"
A: BOTH

**Example 4:**
Q: "What did Apple say about its services revenue?"
A: BOTH

**Example 5:**
Q: "List all the subsidiaries of Microsoft mentioned in its latest 10-K."
A: TEXTUAL

**Example 6:**
Q: "Give me the total R&D expenses over the last three years."
A: TABLE

---

### Now classify this query and return only the classification. NO ADDITIONAL TEXTS:

Q: "{user_query}"

"""

def classify_query(user_query: str)-> str:
    prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
    chain = prompt | chat | StrOutputParser()
    classification = chain.invoke({"user_query":user_query})
    return classification.strip()

if __name__ == "__main__":
    user_query = input("Input the user query: ")
    classification = classify_query(user_query)
    print(classification)