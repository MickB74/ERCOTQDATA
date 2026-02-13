import pandas as pd
import requests
import io
from ercot_queue.fetcher import discover_latest_report_url
from ercot_queue.config import DEFAULT_REPORT_INDEX_URLS

url = discover_latest_report_url(DEFAULT_REPORT_INDEX_URLS[0])
print(f"Downloading: {url}")
response = requests.get(url, timeout=30)

xls = pd.ExcelFile(io.BytesIO(response.content))
print(f"\nAvailable Sheets: {xls.sheet_names}")

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    total_mw = 0
    # Try to find a capacity column in this sheet
    cap_cols = [c for c in df.columns if 'capacity' in str(c).lower() or 'mw' in str(c).lower()]
    if cap_cols:
        total_mw = pd.to_numeric(df[cap_cols[0]], errors='coerce').sum()
    
    print(f"Sheet: {sheet:20} | Rows: {len(df):6} | Est. MW: {total_mw:10,.0f}")
