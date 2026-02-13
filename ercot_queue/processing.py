from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

EMPTY_STRINGS = {"", "na", "n/a", "none", "null", "nan"}


def prepare_queue_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["record_key"])

    df = raw_df.copy()
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [_normalize_column_name(col) for col in df.columns]
    df = _clean_string_values(df)
    df = _infer_numeric_columns(df)
    df = _infer_date_columns(df)

    df["record_key"] = build_record_key(df)
    df = df.drop_duplicates(subset=["record_key"], keep="last").reset_index(drop=True)
    return df


def build_record_key(df: pd.DataFrame) -> pd.Series:
    key_columns = _select_key_columns(df)

    if not key_columns:
        key_columns = list(df.columns[: min(3, len(df.columns))])

    def _row_hash(row: pd.Series) -> str:
        values = [_to_key_string(row.get(col)) for col in key_columns]
        key_material = "||".join(values)
        return hashlib.sha1(key_material.encode("utf-8")).hexdigest()

    return df.apply(_row_hash, axis=1)


def infer_semantic_columns(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)
    return {
        "status": _match_first(columns, [r"status"]),
        "fuel": _match_first(columns, [r"fuel", r"technology", r"resource[_ ]type"]),
        "county": _match_first(columns, [r"county", r"location_county"]),
        "capacity_mw": _match_first(columns, [r"capacity.*mw", r"mw", r"size"]),
        "cod_date": _match_first(
            columns,
            [
                r"proposed.*completion",
                r"commercial.*operation",
                r"cod",
                r"in.?service",
            ],
        ),
        "project_name": _match_first(columns, [r"project", r"name"]),
        "queue_id": _match_first(columns, [r"queue.*(id|number)", r"project.*id", r"^id$"]),
    }


def _select_key_columns(df: pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    priority_patterns = [
        r"queue.*(id|number)",
        r"project.*id",
        r"^id$",
        r"project.*name",
        r"interconnecting.*entity",
        r"resource.*name",
        r"county",
    ]

    key_columns: list[str] = []
    for pattern in priority_patterns:
        match = _match_first(columns, [pattern])
        if match and match not in key_columns:
            key_columns.append(match)
        if len(key_columns) >= 3:
            break

    return key_columns


def _normalize_column_name(column: Any) -> str:
    text = str(column).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "column"


def _clean_string_values(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    for column in object_cols:
        normalized = df[column].astype("string").str.strip()
        lowered = normalized.str.lower()
        df[column] = normalized.mask(lowered.isin(EMPTY_STRINGS), pd.NA)

    return df


def _infer_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_candidates = [
        col
        for col in df.columns
        if re.search(r"(mw|capacity|size|rating|count)$", col) or re.search(r"(mw|capacity|size)", col)
    ]

    for column in numeric_candidates:
        converted = pd.to_numeric(df[column], errors="coerce")
        non_na_ratio = converted.notna().mean() if len(converted) else 0
        if non_na_ratio > 0.6:
            df[column] = converted

    return df


def _infer_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    date_candidates = [
        col
        for col in df.columns
        if re.search(r"date|operation|service|commercial|cod|online", col)
    ]

    for column in date_candidates:
        converted = pd.to_datetime(df[column], errors="coerce")
        non_na_ratio = converted.notna().mean() if len(converted) else 0
        if non_na_ratio > 0.5:
            df[column] = converted

    return df


def _match_first(columns: list[str], patterns: list[str]) -> str | None:
    for pattern in patterns:
        regex = re.compile(pattern)
        for column in columns:
            if regex.search(column):
                return column
    return None


def _to_key_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()
