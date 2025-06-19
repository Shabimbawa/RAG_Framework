import requests
import json
import time
import os
import pandas as pd
import numpy as np
import pdfkit
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import torch
from sentence_transformers import SentenceTransformer
import nltk
#nltk.download("punkt")  needed to be run once only



os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")
print("Current Working Directory:", os.getcwd())

# Opening the OG dataset with all companies in SEC EDGAR
with open("company_tickers_exchange.json", "r") as f:
    CIK_dict = json.load(f)

CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])
nasdaq_df = CIK_df[CIK_df["exchange"] == "Nasdaq"]

# Ticker list so that can find the nasdaq 100 companies since OG json dataset doesn't specify
nasdaq_100_tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'AMD',
    'ABNB', 'AEP', 'AMGN', 'ADI', 'ANSS', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM',
    'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR',
    'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD',
    'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC',
    'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC',
    'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSTR',
    'MDLZ', 'MNST', 'NFLX', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW',
    'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SBUX',
    'SNPS', 'TTWO', 'TMUS', 'TSLA', 'TXN', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY',
    'XEL', 'ZS'
]

# Filtering the Nasdaq 100 companies here
nasdaq_df = nasdaq_df[nasdaq_df["ticker"].isin(nasdaq_100_tickers)].reset_index(drop=True)
# print(nasdaq_df)

cik_list = []

for ticker in nasdaq_100_tickers:
    CIKs = nasdaq_df[nasdaq_df["ticker"]==ticker]["cik"].values
    cik_list.extend(CIKs)

# Making the requests and all from SEC EDGAR
headers = {
    "User-Agent": "rhenzgerard0@gmail.com"
}

# Generating the directories to store the filings
output_dirs = ["10-k_documents", "10-q_documents", "8-k_documents"]

for dir_name in output_dirs:
    os.makedirs(dir_name, exist_ok=True)

# Logic for data from the past 5 years
five_years_prior = datetime.now() - timedelta(5*365)




REGEX_PATTERNS = { #dictionary of annotated keys
    "date": r"\b\d{4}-\d{2}-\d{2}\b",
    "currency": r"\$?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?",
    "percentage": r"\d+\.?\d*%",
    "numeric": r"^-?\d+(?:,\d{3})*(?:\.\d+)?$",
    "ticker": r"\b[A-Z]{1,5}\b"
}


def extract_html_tables(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []

    for table in soup.find_all('table'):
        rows = []
        for tr in table.find_all('tr'):
            row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if any(cell.strip() for cell in row):
                rows.append(row)

        if not rows:
            continue

        max_len = max(len(row) for row in rows)
        padded_rows = [row + [''] * (max_len - len(row)) for row in rows]
        flat_cells = [cell for row in padded_rows for cell in row]
        if all(cell.strip() == '' for cell in flat_cells):
            continue

        columns = [f"col_{i}" for i in range(max_len)]
        df = pd.DataFrame(padded_rows, columns=columns)

        if df.dropna(how="all").shape[0] <= 1:
            continue

        # Find closest non-empty <span> before and after the table
        def get_closest_span(element, direction="prev"):
            if direction == "prev":
                spans = element.find_all_previous("span")
            else:
                spans = element.find_all_next("span")
            for span in spans:
                text = span.get_text(strip=True)
                if text:
                    return text
            return ""

        prev_span = get_closest_span(table, "prev")
        next_span = get_closest_span(table, "next")

        tables.append({
            "table": df,
            "prev_span": prev_span,
            "next_span": next_span
        })

    return tables







# Function to do the actual retrieval of the nasdaq 100 company filings
def filing_retrievals(filtered_filings, cik_stripped, output_dir, form_type, ticker):
    for _, row in filtered_filings.iterrows(): #for each filings, download their htm if doesnt exist yet, extract tables from it always, save to tables_extracted folder, convert to pdf if nonexistant
        file_name = row.primaryDocument
        access_number = row.accessionNumber.replace("-", "")
        html_path = os.path.join(output_dir, f"{cik_stripped}_{file_name}") 
        pdf_path = html_path + ".pdf"

        try:
            if not os.path.exists(html_path):
                # Download and save HTML only if it doesn't exist
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{access_number}/{file_name}"
                req_content = requests.get(filing_url, headers=headers).content.decode("utf-8")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(req_content)
                html_content = req_content
            else:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

            # Extract and save tables regardless of whether PDF exists
            tables = []
            if form_type.upper() in ("10-K", "10-Q"):
                tables = extract_html_tables(html_content)

                form_output_dir = os.path.join("tables_extracted", cik_stripped, form_type.upper())
                os.makedirs(form_output_dir, exist_ok=True)

                base_file_name = file_name.rsplit('.', 1)[0]
                for i, table_info in enumerate(tables):
                    # output_file = f"{base_file_name}_table_{i}.json"
                    # output_path = os.path.join(form_output_dir, output_file)
                    # table_df.to_json(output_path, orient="records", lines=True, index=False)  
                    #remove everything until with open to revert to this ^


                    table_df = table_info["table"]
                    prev_span = table_info["prev_span"]
                    next_span = table_info["next_span"]

                    metadata = {
                        "source_file": file_name,
                        "company_cik": cik_stripped,
                        "form_type": form_type,
                        "company_ticker": ticker,
                        "table_index": i,
                        "extraction_date": datetime.now().isoformat(),
                        "prev_span": prev_span,
                        "next_span": next_span
                    }

                    json_output = {
                        "metadata": metadata,
                        "data": table_df.to_dict(orient="records")
                    }

                    output_file = f"{base_file_name}_table_{i}.json"
                    output_path = os.path.join(form_output_dir, output_file)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(json_output, f, indent=2)

                    print(f"Saved table {i} from {file_name} to {form_output_dir}")
            else:
                print("No tables to be extracted from 8-K form")
            if not tables:
                print(f"No tables found in {file_name}")

            # Skip PDF generation if already done
            if os.path.exists(pdf_path):
                print(f"PDF already exists for {file_name}")
                continue

            # [Optional] PDF generation logic can go here if needed
            else:
                pdfkit.from_file(html_path, pdf_path, options={"quiet": ""})

        except Exception as e:
            print(f"Error with {file_name}: {e}")




# Testing with only 2 for now since we might overload their servers LOL
def download_filings(cik_list, output_dirs, headers, five_years_prior):
    for cik in cik_list[:1]:
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json" #metadata of filings for this CIK

        response = requests.get(url,headers=headers)

        if response.status_code == 200: #if succesful filter by recent, convert these recent to a dataframe
            data = response.json()
            print(f"Data for CIK {cik_padded}")

            recent_filings = pd.DataFrame(data["filings"]["recent"])
            recent_filings["filingDate"] = pd.to_datetime(recent_filings["filingDate"])
            cik_stripped = str(int(cik))

            form_types = {"10-K": output_dirs[0], "10-Q": output_dirs[1], "8-K": output_dirs[2]} 
            for form, output_dir in form_types.items(): #filter df by form types and within 5 years prior
                filtered = recent_filings[(recent_filings["form"] == form) & (recent_filings["filingDate"] >= five_years_prior)]

                if filtered.empty:
                    print(f"No {form}s from the past 5 years for CIK {cik_padded}")
                    continue

                ticker = data.get("tickers", [""])[0] if "tickers" in data else ""
                filing_retrievals(filtered, cik_stripped, output_dir, form, ticker)

            
        else:
            print(f"Failed to fetch data for CIK{cik_padded} (status{response.status_code})")

        time.sleep(0.11) # rate limits


#download_filings(cik_list, output_dirs, headers, five_years_prior)




def extract_and_save_spans(base_path=".", input_folders=["10-k_documents", "10-q_documents", "8-k_documents"]):
    output_base = os.path.join(base_path, "extracted_texts")
    os.makedirs(output_base, exist_ok=True)

    for folder in input_folders:
        input_dir = os.path.join(base_path, folder)

        if not os.path.exists(input_dir):
            print(f"Folder not found: {input_dir}")
            continue

        form_type = folder.replace("_documents", "").lower() + "-texts"
        form_output_dir = os.path.join(output_base, form_type)
        os.makedirs(form_output_dir, exist_ok=True)

        print(f"\nProcessing folder: {input_dir}")

        files = sorted(os.listdir(input_dir))
        for file in files:  # You can change back to `files[:1]` to limit
            if not file.endswith(".htm") and not file.endswith(".html"):
                continue

            # Parse ticker from filename
            try:
                base_name = os.path.splitext(file)[0]
                if "_" in base_name:
                    ticker = base_name.split("_")[1].split("-")[0].lower()
                else:
                    ticker = base_name.split("-")[0].lower()

            except IndexError:
                print(f"Could not parse ticker from filename: {file}")
                continue

            ticker_output_dir = os.path.join(form_output_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)

            file_path = os.path.join(input_dir, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")

                # Remove all tables
                for table in soup.find_all("table"):
                    table.decompose()

                # Extract all non-empty span text
                span_texts = [span.get_text(strip=True) for span in soup.find_all("span") if span.get_text(strip=True)]

                # Save as text file
                output_file = os.path.splitext(file)[0] + ".txt"
                output_path = os.path.join(ticker_output_dir, output_file)

                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write("\n".join(span_texts))

                print(f"Saved: {output_file} ({len(span_texts)} spans) to {ticker_output_dir}")

            except Exception as e:
                print(f"Error processing {file}: {e}")


#extract_and_save_spans()



def segment_dense(base_path=".", input_folders=["10-k_documents", "10-q_documents", "8-k_documents"]):
    segment_config(
        base_path=base_path,
        input_folders=input_folders,
        output_folder="chunked_dense",
        chunk_size=2048,      # ~512 tokens
        chunk_overlap=512
    )


def segment_sparse(base_path=".", input_folders=["10-k_documents", "10-q_documents", "8-k_documents"]):
    segment_config(
        base_path=base_path,
        input_folders=input_folders,
        output_folder="chunked_sparse",
        chunk_size=1024,     # ~256 tokens
        chunk_overlap=256
    )


def segment_config(base_path, input_folders, output_folder, chunk_size, chunk_overlap):
    input_base = os.path.join(base_path, "extracted_texts")
    output_base = os.path.join(base_path, output_folder)
    os.makedirs(output_base, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for folder in input_folders:
        form_type = folder.replace("_documents", "").lower() + "-texts"
        input_dir = os.path.join(input_base, form_type)

        if not os.path.exists(input_dir):
            print(f"Input folder not found: {input_dir}")
            continue

        output_form_dir = os.path.join(output_base, form_type)
        os.makedirs(output_form_dir, exist_ok=True)

        print(f"\nProcessing {form_type}...")

        for ticker in sorted(os.listdir(input_dir)):  # limit to first ticker
            ticker_input_dir = os.path.join(input_dir, ticker)
            if not os.path.isdir(ticker_input_dir):
                continue

            ticker_output_dir = os.path.join(output_form_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)

            for file in sorted(os.listdir(ticker_input_dir)):  # limit to first 3 files
                if not file.endswith(".txt"):
                    continue

                file_path = os.path.join(ticker_input_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        full_text = f.read()

                    chunks = splitter.split_text(full_text)
                    chunk_objects = []
                    char_pointer = 0

                    for i, chunk in enumerate(chunks):
                        start = full_text.find(chunk, char_pointer)
                        end = start + len(chunk)
                        char_pointer = end  # move pointer forward

                        chunk_objects.append({
                            "chunk_id": i + 1,
                            "content": chunk,
                            "start_index": start,
                            "end_index": end,
                            "source_file": file,
                            "ticker": ticker,
                            "form_type": form_type.replace("-texts", "")  
                        })

                    output_filename = os.path.splitext(file)[0] + "_chunks.json"
                    output_path = os.path.join(ticker_output_dir, output_filename)

                    with open(output_path, "w", encoding="utf-8") as json_file:
                        json.dump(chunk_objects, json_file, indent=2, ensure_ascii=False)

                    print(f"Chunked and saved: {file} -> {len(chunks)} chunks")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

#segment_dense()
#segment_sparse()


def embed_chunks_bm25(base_path=".", input_folder="chunked_sparse", output_folder="embedded_sparse_bm25"):
    input_base = os.path.join(base_path, input_folder)
    output_base = os.path.join(base_path, output_folder)
    os.makedirs(output_base, exist_ok=True)

    for form_type in os.listdir(input_base):
        form_input_dir = os.path.join(input_base, form_type)
        form_output_dir = os.path.join(output_base, form_type)
        os.makedirs(form_output_dir, exist_ok=True)

        for ticker in os.listdir(form_input_dir):
            ticker_input_dir = os.path.join(form_input_dir, ticker)
            ticker_output_dir = os.path.join(form_output_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)

            for file in sorted(os.listdir(ticker_input_dir)):
                output_file = file.replace("_chunks.json", "_embedded.json")
                output_path = os.path.join(ticker_output_dir, output_file)
                if os.path.exists(output_path):
                    print(f"Skipping (already embedded): {file}")
                    continue
                if not file.endswith("_chunks.json"):
                    continue

                file_path = os.path.join(ticker_input_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                tokenized_corpus = [word_tokenize(c["content"].lower(), preserve_line=True) for c in chunks]

                bm25 = BM25Okapi(tokenized_corpus)

                embedded = []
                for i, chunk in enumerate(chunks):
                    tokens = tokenized_corpus[i]
                    scores = bm25.get_scores(tokens)
                    embedded.append({
                        "chunk_id": chunk["chunk_id"],
                        "bm25_scores": list(scores),
                        "metadata": {
                            "source_file": chunk["source_file"],
                            "ticker": chunk["ticker"],
                            "form_type": chunk["form_type"],
                            "start_index": chunk["start_index"],
                            "end_index": chunk["end_index"]
                        }
                    })


                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(embedded, out, indent=2, ensure_ascii=False)

                print(f"BM25 embedded: {file} -> {len(embedded)} chunks")


def embed_chunks_finbert(base_path=".", input_folder="chunked_dense", output_folder="embedded_dense_finbert", batch_size=256):

    # Create output directories
    input_base = os.path.join(base_path, input_folder)
    output_base = os.path.join(base_path, output_folder)
    os.makedirs(output_base, exist_ok=True)

    # Load model with optimizations
    model = SentenceTransformer("yiyanghkust/finbert-tone")
    
    # FP16 conversion and GPU acceleration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.half().to(device) if device == "cuda" else model

    # Process each filing type (10-K, 10-Q, etc.)
    for form_type in os.listdir(input_base):
        form_input_dir = os.path.join(input_base, form_type)
        form_output_dir = os.path.join(output_base, form_type)
        os.makedirs(form_output_dir, exist_ok=True)

        # Process each company (ticker)
        for ticker in os.listdir(form_input_dir)[:1]:
            ticker_input_dir = os.path.join(form_input_dir, ticker)
            ticker_output_dir = os.path.join(form_output_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)

            # Process each chunk file
            for file in sorted(os.listdir(ticker_input_dir))[:3]:
                if not file.endswith("_chunks.json"):
                    continue

                file_path = os.path.join(ticker_input_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                texts = [chunk["content"] for chunk in chunks]
                
                # Generate embeddings with optimizations
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device=device
                )

                # Prepare embedded data structure
                embedded = []
                for chunk, emb in zip(chunks, embeddings):
                    # Convert tensor to list if needed
                    if hasattr(emb, 'cpu'):
                        emb = emb.cpu().numpy().tolist()
                    else:
                        emb = emb.tolist()
                        
                    embedded.append({
                        "chunk_id": chunk["chunk_id"],
                        "embedding": emb,
                        "metadata": {
                            "source_file": chunk.get("source_file", ""),
                            "ticker": chunk.get("ticker", ""),
                            "form_type": chunk.get("form_type", ""),
                            "start_index": chunk.get("start_index", None),
                            "end_index": chunk.get("end_index", None)
                        }
                    })

                # Save results
                output_file = file.replace("_chunks.json", "_embedded.json")
                output_path = os.path.join(ticker_output_dir, output_file)
                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(embedded, out, indent=2, ensure_ascii=False)

                print(f"FinBERT embedded: {file} -> {len(embedded)} chunks")

    print("Embedding process completed.")


embed_chunks_bm25()
#embed_chunks_finbert()
