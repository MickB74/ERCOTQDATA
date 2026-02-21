# CLAUDE.md

## Project Overview

**ERCOT Interconnection Queue Explorer** — a Streamlit web application that fetches the ERCOT (Electric Reliability Council of Texas) interconnection queue, tracks row-level changes between refreshes, and visualizes the data interactively.

## Tech Stack

- **Language:** Python 3.11+
- **UI:** Streamlit
- **Data:** pandas, pyarrow (parquet), plotly
- **Fetching:** requests, beautifulsoup4, lxml, openpyxl, pypdf

## Project Layout

```
app.py                        # Streamlit UI (entry point)
ercot_queue/
  fetcher.py                  # Data retrieval from ERCOT MIS/GIS sources
  processing.py               # Column normalization and key generation
  diffing.py                  # Refresh-to-refresh change detection
  store.py                    # Local snapshot and metadata persistence
  validation.py               # Validation against external sources
  config.py                   # Configuration constants
tests/
  test_diffing.py
  test_store.py
data/
  snapshots/                  # Per-refresh parquet snapshots
  changes/                    # JSON diff reports
  current_snapshot.parquet    # Always-current local snapshot
  metadata.json               # Snapshot index/history
```

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then click **Refresh Data** in the sidebar to pull the latest ERCOT data.

## Running Tests

```bash
python -m unittest discover -s tests -v
```

No pytest or linting config is currently set up.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `ERCOT_REPORT_INDEX_URL` | (built-in) | Override the MIS index URL |
| `ERCOT_DATA_PRODUCT_URL` | (built-in) | Override the data-product-details URL |
| `ERCOT_REQUEST_TIMEOUT` | `60` | HTTP timeout in seconds |
| `ERCOT_MAX_CHANGE_SAMPLE_ROWS` | `500` | Max rows stored per change report |

## Key Architecture Notes

- Auto-fetch starts at ERCOT data product page `PG7-200-ER`, derives a `Report Type ID`, then pulls the latest `GIS_Report_*` file from ERCOT MIS.
- Snapshots are written as parquet with UTC timestamps; refreshes are skipped when the online report matches the current local snapshot (hash check).
- The diffing layer detects added, removed, and field-level changed rows between snapshots.
- Views: **Interconnection Queue**, **Generation Fleet** (MORA data), **Building Interconnects** (GIS text-match candidates).
- All timestamps are stored in UTC.

## Development Notes

- The active development branch is `claude/update-claude-md-vLT9N`.
- `data/` directories are gitignored (snapshots, metadata, change reports).
- No CI/CD pipelines are currently configured.
