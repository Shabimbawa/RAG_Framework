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

# os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")
# print("Current Working Directory:", os.getcwd())

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
def filing_retrievals(filtered_filings, cik_stripped, output_dir, form_type):
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
                        "table_index": i,
                        "extraction_date": datetime.now().isoformat(),
                        "column_headers": table_df.columns.tolist(),
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

                filing_retrievals(filtered, cik_stripped, output_dir, form)
            
        else:
            print(f"Failed to fetch data for CIK{cik_padded} (status{response.status_code})")

        time.sleep(0.11) # rate limits


download_filings(cik_list, output_dirs, headers, five_years_prior)
