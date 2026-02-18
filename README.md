# ERCOT Interconnection Queue Explorer

Interactive Streamlit app to pull the ERCOT interconnection queue, explore it with filters/charts, refresh data on demand, and track exactly what changed between refreshes.

## Features

- Pull latest queue data from ERCOT sources (or from a custom ERCOT file URL)
- Persist snapshots to parquet on every refresh with UTC pull timestamp
- Detect and report row-level changes:
  - added projects
  - removed projects
  - changed field values
- Skip refresh writes when the online ERCOT report matches the current local snapshot
- Filter queue records by status, fuel/technology, county, capacity, and COD date range
- Visualize queue data with charts:
  - capacity/projects by status
  - projects by fuel/technology
  - cumulative COD timeline
- Generation Fleet view with:
  - MORA fleet data from ERCOT Resource Details
  - In-app refresh controls for MORA fleet data
- Building Interconnects view with:
  - Candidate rows from ERCOT GIS snapshot text matches (data centers, offices, load/campus/company terms)
  - In-app refresh controls and filtered candidate table/chart exports
- Download filtered records as CSV
- View snapshot history with per-refresh diff summary
- Validate against an independent external source (Interconnection.fyi) with mismatch reports

## Project Layout

- `app.py`: Streamlit UI
- `ercot_queue/fetcher.py`: data retrieval logic
- `ercot_queue/processing.py`: column normalization and key generation
- `ercot_queue/diffing.py`: refresh-to-refresh change detection
- `ercot_queue/store.py`: local snapshot and metadata persistence
- `data/snapshots/`: stored queue snapshots
- `data/current_snapshot.parquet`: always-current local snapshot file
- `data/changes/`: stored diff reports
- `data/metadata.json`: snapshot index/history

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. In the app sidebar, click `Refresh Data`.

## Notes

- Auto-fetch starts at ERCOT data product page `PG7-200-ER`, derives `Report Type ID`, then pulls the latest GIS file from ERCOT MIS.
- GIS file selection explicitly prefers the latest `GIS_Report_*` entry (for example `GIS_Report_January2026`) over non-GIS companion files.
- Excel ingestion processes all workbook tabs and records tab names/count in snapshot metadata.
- If auto-discovery fails, paste an ERCOT `data-product-details` URL or direct ERCOT CSV/XLS/XLSX/ZIP URL into `ERCOT Source URL` and refresh.
- All pull timestamps are stored in UTC.

## Optional Environment Variables

- `ERCOT_REPORT_INDEX_URL`: override the default MIS index URL
- `ERCOT_DATA_PRODUCT_URL`: override the default ERCOT data-product-details URL
- `ERCOT_REQUEST_TIMEOUT`: HTTP timeout in seconds (default `60`)
- `ERCOT_MAX_CHANGE_SAMPLE_ROWS`: limit stored sample rows in change reports (default `500`)
