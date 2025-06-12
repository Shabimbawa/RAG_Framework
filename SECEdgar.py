import requests
import json
import time
import os
import pandas as pd
import numpy as np
import pdfkit
import time
from datetime import datetime, timedelta

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
    "User-Agent": "rhenzlargo80@gmail.com"
}

# Generating the directories to store the filings
output_dirs = ["10-k_documents", "10-q_documents", "8-k_documents"]

for dir_name in output_dirs:
    os.makedirs(dir_name, exist_ok=True)

# Logic for data from the past 5 years
five_years_prior = datetime.now() - timedelta(5*365)

# Function to do the actual retrieval of the nasdaq 100 company filings
def filing_retrievals(filtered_filings, cik_stripped, output_dir):
    for _, row in filtered_filings.iterrows():
        file_name = row.primaryDocument
        access_number = row.accessionNumber.replace("-", "")
        html_path = os.path.join(output_dir, f"{cik_stripped}_{file_name}") 
        pdf_path = html_path + ".pdf"

        try:
            if os.path.exists(pdf_path):
                print(f"PDF already exists for {file_name}") # so that it doesn't re-download files
                continue # exits the current 'for' loop iteration

            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{access_number}/{file_name}"
            req_content = requests.get(filing_url, headers=headers).content.decode("utf-8")

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(req_content)

            pdfkit.from_file(html_path, pdf_path, options={"quiet": ""})

        except Exception as e:
            print(f"Error with {file_name}: {e}")

# Testing with only 2 for now since we might overload their servers LOL
for cik in cik_list:
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    response = requests.get(url,headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"Data for CIK {cik_padded}")

        recent_filings = pd.DataFrame(data["filings"]["recent"])
        recent_filings["filingDate"] = pd.to_datetime(recent_filings["filingDate"])
        cik_stripped = str(int(cik))

        form_types = {"10-K": output_dirs[0], "10-Q": output_dirs[1], "8-K": output_dirs[2]} 
        for form, output_dir in form_types.items():
            filtered = recent_filings[(recent_filings["form"] == form) & (recent_filings["filingDate"] >= five_years_prior)]

            if filtered.empty:
                print(f"No {form}s from the past 5 years for CIK {cik_padded}")
                continue

            filing_retrievals(filtered, cik_stripped, output_dir)
           
    else:
        print(f"Failed to fetch data for CIK{cik_padded} (status{response.status_code})")

    time.sleep(0.11) # time delay to reduce number of reqs to SEC EDGAR


