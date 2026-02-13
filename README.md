# ERCOT Interconnection Queue Explorer

Interactive Streamlit app to pull the ERCOT interconnection queue, explore it with filters/charts, refresh data on demand, and track exactly what changed between refreshes.

## Features

- Pull latest queue data from ERCOT sources (or from a custom file URL)
- Snapshot every refresh with UTC pull timestamp
- Detect and report row-level changes:
  - added projects
  - removed projects
  - changed field values
- Filter queue records by status, fuel/technology, county, capacity, and COD date range
- Visualize queue data with charts:
  - capacity/projects by status
  - projects by fuel/technology
  - cumulative COD timeline
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

- Auto-fetch first tries `gridstatus`, then attempts ERCOT MIS report link discovery.
- If your environment blocks one or both methods, paste a direct CSV/XLS/XLSX/ZIP file URL into `Custom ERCOT file URL` and refresh.
- All pull timestamps are stored in UTC.

## Optional Environment Variables

- `ERCOT_REPORT_INDEX_URL`: override the default MIS index URL
- `ERCOT_REQUEST_TIMEOUT`: HTTP timeout in seconds (default `60`)
- `ERCOT_MAX_CHANGE_SAMPLE_ROWS`: limit stored sample rows in change reports (default `500`)
