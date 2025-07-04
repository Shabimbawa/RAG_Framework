import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

def extract_html_tables(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []

    item_regex = re.compile(r'ITEM\s+(\d+[A-Z]?)\.?\s*[:\-]?\s*(.*)', re.IGNORECASE)
    current_section = None

    def clean_text(text):
        return text.replace('\xa0', ' ').strip()

    def merge_currency_cells(row):
        merged = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            if row[i] == "$" and i + 1 < len(row):
                merged.append(f"${row[i + 1]}")
                skip = True
            else:
                merged.append(row[i])
        return merged

    def merge_percent_cells(row):
        merged = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            current = row[i].strip()
            next_cell = row[i + 1].strip() if i + 1 < len(row) else ""
            if re.match(r"^-?\(?\d+(\.\d+)?\)?$", current) and next_cell == "%":
                merged.append(f"{current}%")
                skip = True
            else:
                merged.append(current)
        return merged

    def get_closest_span(element, direction="prev"):
        spans = element.find_all_previous("span") if direction == "prev" else element.find_all_next("span")
        for span in spans:
            text = clean_text(span.get_text())
            if text:
                return text
        return ""

    def is_width_only_row(tr):
        tds = tr.find_all("td")
        return all(td.get("style", "").startswith("width") and not td.get_text(strip=True) for td in tds)

    def get_table_header_and_data(tr_elements):
        header_row = None
        data_rows = []

        for tr in tr_elements:
            if is_width_only_row(tr):
                continue

            cells = tr.find_all(["td", "th"])
            texts = [clean_text(td.get_text()) for td in cells]

            # Header row = row with bold spans
            if not header_row and any("font-weight:700" in span.get("style", "") for td in cells for span in td.find_all("span")):
                header_row = texts
                continue

            row = merge_currency_cells(texts)
            row = merge_percent_cells(row)
            if any(cell.strip() for cell in row):
                data_rows.append(row)

        return header_row, data_rows

    for tag in soup.find_all(["span", "table"]):
        text = tag.get_text(strip=True)

        # Update section header
        match = item_regex.search(text)
        if match:
            number = match.group(1)
            title = match.group(2)
            current_section = f"Item {number}: {title}" if title else f"Item {number}"

        if tag.name == "table":
            trs = tag.find_all("tr")
            header, data_rows = get_table_header_and_data(trs)

            if not data_rows:
                continue

            # If header exists and first cell is empty, call it "Category"
            if header and not header[0].strip():
                header[0] = "Category"

            # Left-shift header by removing empty cells
            header = [cell.strip() for cell in header if cell.strip()] if header else []
            max_cols = max(len(header), max((len(row) for row in data_rows), default=0))

            # Pad header
            while len(header) < max_cols:
                header.append(f"col_{len(header)}")

            # Left-shift and pad data rows
            padded_data = []
            for row in data_rows:
                shifted_row = [cell.strip() for cell in row if cell.strip()]
                while len(shifted_row) < max_cols:
                    shifted_row.append("")
                padded_data.append(shifted_row)

            df = pd.DataFrame(padded_data, columns=header)

            if df.dropna(how="all").shape[0] <= 1:
                continue

            prev_span = get_closest_span(tag, "prev")
            next_span = get_closest_span(tag, "next")

            tables.append({
                "table": df,
                "prev_span": prev_span,
                "next_span": next_span,
                "section_header": current_section or "Unknown",
                "num_rows": len(df),
                "num_cols": len(df.columns)
            })

    return tables