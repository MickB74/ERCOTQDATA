from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd


def calculate_diff(
    previous_df: pd.DataFrame | None,
    current_df: pd.DataFrame,
    *,
    max_sample_rows: int = 500,
) -> dict[str, Any]:
    if previous_df is None or previous_df.empty:
        return {
            "summary": {
                "added": int(len(current_df)),
                "removed": 0,
                "changed": 0,
                "unchanged": 0,
            },
            "added_sample": _rows_for_keys(current_df, list(current_df.get("record_key", [])), max_sample_rows),
            "removed_sample": [],
            "changed_records": [],
            "changed_field_details": [],
        }

    prev = _ensure_unique_keys(previous_df)
    curr = _ensure_unique_keys(current_df)

    previous_keys = set(prev["record_key"])
    current_keys = set(curr["record_key"])

    added_keys = sorted(current_keys - previous_keys)
    removed_keys = sorted(previous_keys - current_keys)
    common_keys = sorted(previous_keys & current_keys)

    comparison_columns = sorted(
        set(prev.columns).union(curr.columns).difference({"record_key"})
    )

    prev_map = prev.set_index("record_key", drop=False).to_dict(orient="index")
    curr_map = curr.set_index("record_key", drop=False).to_dict(orient="index")

    changed_records: list[dict[str, Any]] = []
    changed_field_details: list[dict[str, Any]] = []

    for record_key in common_keys:
        prev_row = prev_map[record_key]
        curr_row = curr_map[record_key]

        row_changes: list[str] = []
        for column in comparison_columns:
            old_value = _normalize_value(prev_row.get(column))
            new_value = _normalize_value(curr_row.get(column))
            if old_value == new_value:
                continue

            row_changes.append(column)
            if len(changed_field_details) < max_sample_rows:
                changed_field_details.append(
                    {
                        "record_key": record_key,
                        "field": column,
                        "old": _json_safe(prev_row.get(column)),
                        "new": _json_safe(curr_row.get(column)),
                    }
                )

        if row_changes:
            changed_records.append(
                {
                    "record_key": record_key,
                    "changed_fields": row_changes,
                    "changed_field_count": len(row_changes),
                }
            )

    unchanged = len(common_keys) - len(changed_records)
    return {
        "summary": {
            "added": len(added_keys),
            "removed": len(removed_keys),
            "changed": len(changed_records),
            "unchanged": unchanged,
        },
        "added_sample": _rows_for_keys(curr, added_keys, max_sample_rows),
        "removed_sample": _rows_for_keys(prev, removed_keys, max_sample_rows),
        "changed_records": changed_records[:max_sample_rows],
        "changed_field_details": changed_field_details,
    }


def _ensure_unique_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "record_key" not in df.columns:
        raise ValueError("Dataframe must include a record_key column")
    return df.drop_duplicates(subset=["record_key"], keep="last")


def _rows_for_keys(df: pd.DataFrame, keys: Iterable[str], limit: int) -> list[dict[str, Any]]:
    key_set = set(list(keys)[:limit])
    if not key_set:
        return []

    subset = df[df["record_key"].isin(key_set)]
    return [_jsonify_record(record) for record in subset.to_dict(orient="records")]


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value).strip()


def _jsonify_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(value) for key, value in record.items()}


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    return value
