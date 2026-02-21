from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
CHANGE_DIR = DATA_DIR / "changes"
METADATA_PATH = DATA_DIR / "metadata.json"
CURRENT_SNAPSHOT_PATH = DATA_DIR / "current_snapshot.parquet"

DEFAULT_DATA_PRODUCT_URLS = [
    "https://www.ercot.com/mp/data-products/data-product-details?id=pg7-200-er",
]

DEFAULT_REPORT_INDEX_URLS = [
    "https://www.ercot.com/misapp/GetReports.do?reportTypeId=15933&reportTitle=Generation%20Interconnection%20Status%20Report&showHTMLView=&mimicKey",
    "http://www.ercot.com/misapp/GetReports.do?reportTypeId=15933&reportTitle=Generation%20Interconnection%20Status%20Report&showHTMLView=&mimicKey",
]

def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        import warnings
        warnings.warn(f"Invalid value for {name}={raw!r}; using default {default}", stacklevel=2)
        return default


REQUEST_TIMEOUT_SECONDS = _parse_int_env("ERCOT_REQUEST_TIMEOUT", 60)
MAX_CHANGE_SAMPLE_ROWS = _parse_int_env("ERCOT_MAX_CHANGE_SAMPLE_ROWS", 500)
