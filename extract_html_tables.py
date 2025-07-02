import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from collections import OrderedDict

def extract_html_tables(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []

    item_regex = re.compile(r'ITEM\s+(\d+[A-Z]?)\.?\s*[:\-]?\s*(.*)', re.IGNORECASE)
    current_section = None

    def clean_text(text):
        if not text:
            return ""

        replacements = {
            "\xa0": " ",  # non-breaking space
            "\u200b": " ",  # zero-width space
            "\x91": "‘", "\u2018": "‘",
            "\x92": "’", "\u2019": "’",
            "\x93": "“", "\u201c": "“",
            "\x94": "”", "\u201d": "”",
            "\x95": "•",
            "\x96": "-", "\x97": "-", "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
            "\x98": "˜",
            "\x99": "™",
            "\u2009": " ",
            "\u00ae": "®",
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_non_data_row(row):
        return all(not re.search(r"[\d\$\%]", cell) for cell in row)

    def merge_header_rows(row1, row2):
        return [f"{a.strip()} {b.strip()}".strip() for a, b in zip(row1, row2)]

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
    
    def get_cells_with_colspan(tr):
        cells = []
        for cell in tr.find_all(['td', 'th']):
            text = clean_text(cell.get_text())
            colspan = int(cell.get("colspan", 1))
            cells.extend([text] * colspan)
        return cells
        
    def deduplicate_header(header):
        seen = OrderedDict()
        for i, h in enumerate(header):
            if h not in seen:
                seen[h] = f"{h}" if h else f"col_{i}"
            else:
                # If duplicate, add suffix like "_1", "_2"
                count = sum(k.startswith(h) for k in seen.keys())
                seen[f"{h}_{count}"] = h
        return list(seen.keys())

    def get_closest_span(element, direction="prev"):
        spans = element.find_all_previous("span") if direction == "prev" else element.find_all_next("span")
        for span in spans:
            text = clean_text(span.get_text())
            if text:
                return text
        return ""

    for tag in soup.find_all(["span", "table"]):
        text = tag.get_text(strip=True)
        match = item_regex.search(text)
        if match:
            number = match.group(1)
            title = match.group(2)
            current_section = f"Item {number}: {title}" if title else f"Item {number}"

        if tag.name == "table":
            raw_rows = []
            for tr in tag.find_all("tr"):
                cell_objs = get_cells_with_colspan(tr)
                row = []
                max_col = max([c["col_index"] + c["colspan"] for c in cell_objs]) if cell_objs else 0
                row = [""] * max_col
                for cell in cell_objs:
                    row[cell["col_index"]] = cell["text"]
                row = merge_currency_cells(row)
                row = merge_percent_cells(row)
                if any(cell.strip() for cell in row):
                    raw_rows.append(row)

            if not raw_rows:
                continue

            max_len = max(len(r) for r in raw_rows)
            padded_rows = [r + [""] * (max_len - len(r)) for r in raw_rows]

            header_row = None
            data_start_index = 0

            if len(padded_rows) >= 2:
                if is_non_data_row(padded_rows[0]) and is_non_data_row(padded_rows[1]):
                    header_row = merge_header_rows(padded_rows[0], padded_rows[1])
                    data_start_index = 2

            if header_row is None and len(padded_rows) >= 1:
                if is_non_data_row(padded_rows[0]):
                    header_row = padded_rows[0]
                    data_start_index = 1

            if header_row:
                # Clean and validate header
                header_row = [h.strip() if h.strip() else f"col_{i}" for i, h in enumerate(header_row)]
                header_row = deduplicate_header(header_row)

                if not header_row[0]: header_row[0] = "Category"
                # Ensure header matches data dimensions
                if len(header_row) < max_len:
                    header_row += [f"col_{i}" for i in range(len(header_row), max_len)]
                elif len(header_row) > max_len:
                    header_row = header_row[:max_len]
                
                
            else:
                header_row = [f"col_{i}" for i in range(max_len)]


            data_rows = padded_rows[data_start_index:]

            df = pd.DataFrame(data_rows, columns=header_row)
            df = df.loc[:, (df != '').any(axis=0)]
            df = df.dropna(how='all')

            if len(df) <= 1:
                continue

            df = df.dropna(axis=1, how='all')

            if df.shape[1] == 0:
                continue

            prev_span = get_closest_span(tag, "prev")
            next_span = get_closest_span(tag, "next")

            if not df.empty:
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