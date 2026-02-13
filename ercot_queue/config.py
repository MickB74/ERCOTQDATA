from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
CHANGE_DIR = DATA_DIR / "changes"
METADATA_PATH = DATA_DIR / "metadata.json"

DEFAULT_REPORT_INDEX_URLS = [
    "https://mis.ercot.com/misapp/GetReports.do?reportTypeId=15933&reportTitle=Generation%20Interconnection%20Status%20Report&showHTMLView=&mimicKey",
    "http://mis.ercot.com/misapp/GetReports.do?reportTypeId=15933&reportTitle=Generation%20Interconnection%20Status%20Report&showHTMLView=&mimicKey",
]

REQUEST_TIMEOUT_SECONDS = int(os.getenv("ERCOT_REQUEST_TIMEOUT", "60"))
MAX_CHANGE_SAMPLE_ROWS = int(os.getenv("ERCOT_MAX_CHANGE_SAMPLE_ROWS", "500"))
