from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ercot_queue.config import CHANGE_DIR, DATA_DIR, METADATA_PATH, SNAPSHOT_DIR


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    CHANGE_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata() -> dict[str, Any]:
    ensure_data_dirs()
    if not METADATA_PATH.exists():
        return {"snapshots": []}

    with METADATA_PATH.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_metadata(metadata: dict[str, Any]) -> None:
    ensure_data_dirs()
    with METADATA_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)


def save_snapshot(
    df: pd.DataFrame,
    *,
    source_metadata: dict[str, Any],
    diff_report: dict[str, Any],
) -> dict[str, Any]:
    ensure_data_dirs()

    pulled_at = datetime.now(timezone.utc)
    snapshot_id = pulled_at.strftime("%Y%m%dT%H%M%SZ")

    snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}.csv"
    change_path = CHANGE_DIR / f"{snapshot_id}.json"

    serializable_df = _serialize_dataframe(df)
    serializable_df.to_csv(snapshot_path, index=False)

    with change_path.open("w", encoding="utf-8") as file_handle:
        json.dump(diff_report, file_handle, indent=2)

    entry = {
        "snapshot_id": snapshot_id,
        "pulled_at_utc": pulled_at.isoformat(),
        "row_count": int(len(df)),
        "snapshot_path": str(snapshot_path),
        "change_path": str(change_path),
        "source": source_metadata.get("source", "unknown"),
        "source_url": source_metadata.get("source_url"),
        "diff_summary": diff_report.get("summary", {}),
    }

    metadata = load_metadata()
    snapshots = metadata.setdefault("snapshots", [])
    snapshots.append(entry)
    snapshots.sort(key=lambda item: item["snapshot_id"])
    save_metadata(metadata)

    return entry


def load_latest_snapshot() -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    metadata = load_metadata()
    snapshots = metadata.get("snapshots", [])
    if not snapshots:
        return None, None

    latest = snapshots[-1]
    path = Path(latest["snapshot_path"])
    if not path.exists():
        return None, latest

    df = pd.read_csv(path, low_memory=False)
    df = _restore_dtypes(df)
    return df, latest


def load_change_report(snapshot_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot_meta:
        return None

    change_path = snapshot_meta.get("change_path")
    if not change_path:
        return None

    path = Path(change_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_snapshot_history(limit: int = 20) -> list[dict[str, Any]]:
    metadata = load_metadata()
    snapshots = metadata.get("snapshots", [])
    return list(reversed(snapshots[-limit:]))


def _serialize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    serializable = df.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
    return serializable


def _restore_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if "date" in column or "operation" in column or "service" in column or "cod" in column:
            converted = pd.to_datetime(df[column], errors="coerce")
            non_na_ratio = converted.notna().mean() if len(converted) else 0
            if non_na_ratio > 0.5:
                df[column] = converted

        if "mw" in column or "capacity" in column:
            converted_num = pd.to_numeric(df[column], errors="coerce")
            non_na_ratio = converted_num.notna().mean() if len(converted_num) else 0
            if non_na_ratio > 0.6:
                df[column] = converted_num

    return df
