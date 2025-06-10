import requests
import json
import time
import os
import pandas as pd
import numpy as np
import pdfkit


# New folder to store retreived data from SEC EDGAR
output_dir = "sec_submissions"
os.makedirs(output_dir, exist_ok=True)

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

# Testing with only 2 for now since we might overload their servers LOL
for cik in cik_list[:2]:
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    response = requests.get(url,headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"Data for CIK {cik_padded}")

        # file_path = os.path.join(output_dir, f"{cik_padded}.json")
        # with open(file_path, "w") as f:
        #      json.dump(data, f, indent=4)

    else:
        print(f"Failed to fetch data for CIK{cik_padded} (status{response.status_code})")

    time.sleep(0.11)


# Might transfer code below into the loop above in order to make it dynamic
with open("sec_submissions/0000320193.json", "r") as f: 
    Apple_data = json.load(f)

Apple_data = pd.DataFrame(Apple_data["filings"]["recent"])

access_number = Apple_data[Apple_data.form == "10-K"].accessionNumber.values[0].replace("-", "")
file_name = Apple_data[Apple_data.form == "10-K"].primaryDocument.values[0]

url = f"https://www.sec.gov/Archives/edgar/data/0000320193/{access_number}/{file_name}"
req_content = requests.get(url, headers=headers).content.decode("utf-8")

output_dir_10k = "10-k_documents"
os.makedirs(output_dir_10k, exist_ok=True)

html_path = os.path.join(output_dir_10k, file_name)
pdf_path = html_path + ".pdf"

with open(html_path, "w", encoding="utf-8") as f:
    f.write(req_content)

pdfkit.from_file(html_path, pdf_path, options={"quiet": ""})