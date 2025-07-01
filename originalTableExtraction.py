
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

    def is_dual_header(rows):
        if len(rows) < 2:
            return False
        row1 = rows[0]
        row2 = rows[1]
        if any(cell.strip().startswith("$") for cell in row2):
            return False

        
        return True 


    def get_closest_span(element, direction="prev"):
        spans = element.find_all_previous("span") if direction == "prev" else element.find_all_next("span")
        for span in spans:
            text = clean_text(span.get_text())
            if text:
                return text
        return ""
    def merge_percent_cells(row):
        merged = []
        skip = False
        for i in range(len(row)):
            if skip:
                skip = False
                continue
            current = row[i].strip()
            next_cell = row[i + 1].strip() if i + 1 < len(row) else ""

            # If this is a number and next is '%', merge
            if re.match(r"^-?\(?\d+(\.\d+)?\)?$", current) and next_cell == "%":
                merged.append(f"{current}%")
                skip = True
            else:
                merged.append(current)
        return merged
    
    for tag in soup.find_all(["span", "table"]):
        text = tag.get_text(strip=True)

        # Update section header
        match = item_regex.search(text)
        if match:
            number = match.group(1)
            title = match.group(2)
            current_section = f"Item {number}: {title}" if title else f"Item {number}"

        if tag.name == "table":
            rows = []
            for tr in tag.find_all("tr"):
                row = [clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
                row = merge_currency_cells(row)
                row = merge_percent_cells(row)
                if any(cell.strip() for cell in row):
                    rows.append(row)

            if not rows:
                continue

            max_len = max(len(row) for row in rows)
            padded_rows = [row + [''] * (max_len - len(row)) for row in rows]

            if all(cell.strip() == '' for row in padded_rows for cell in row):
                continue

            df = None

            if is_dual_header(padded_rows):
                header_row = [f"{a.strip()} {b.strip()}".strip() for a, b in zip(padded_rows[0], padded_rows[1])]
                if not header_row[0]:
                    header_row[0] = "Category"
                data_rows = padded_rows[2:]
                df = pd.DataFrame(data_rows, columns=header_row)
            else:
                first_row = padded_rows[0]
                is_header_row = all(len(cell) < 40 for cell in first_row)
                if is_header_row:
                    if not first_row[0]:
                        first_row[0] = "Category"
                    column_headers = [
                        col.strip() if col.strip() else f"col_{i}"
                        for i, col in enumerate(first_row)
                    ]
                    data_rows = padded_rows[1:]
                    df = pd.DataFrame(data_rows, columns=column_headers)
                else:
                    df = pd.DataFrame(padded_rows, columns=[f"col_{i}" for i in range(max_len)])

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












