import os
import pandas as pd
from ercot_queue.fetcher import discover_latest_report_url, _download_table
from ercot_queue.config import DEFAULT_REPORT_INDEX_URLS

# Force manual fetch by bypassing gridstatus
url = discover_latest_report_url(DEFAULT_REPORT_INDEX_URLS[0])
print(f"Manual Fetching from: {url}")
df = _download_table(url)

print(f"\nManual Fetch Row Count: {len(df)}")
if '_source_sheet' in df.columns:
    print("Sheets included:", df['_source_sheet'].unique())

try:
    import gridstatus
    iso = gridstatus.Ercot()
    gs_df = iso.get_interconnection_queue()
    print(f"GridStatus Row Count: {len(gs_df)}")
except Exception as e:
    print(f"GridStatus failed: {e}")
