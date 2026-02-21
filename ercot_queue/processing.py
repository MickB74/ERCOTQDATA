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
    df = _dedupe_column_names(df)
    df = _clean_string_values(df)
    df = _infer_numeric_columns(df)
    df = _infer_date_columns(df)
    df = _remap_storage_fuel(df)

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
    return {
        "status": _select_semantic_column(
            df,
            preferred_exact=["project_status", "gim_study_phase", "status"],
            patterns=[r"project_status", r"gim.*phase", r"status", r"milestone"],
        ),
        "fuel": _select_semantic_column(
            df,
            preferred_exact=["fuel", "technology", "resource_type"],
            patterns=[r"fuel", r"technology", r"resource[_ ]type"],
        ),
        "technology": _select_semantic_column(
            df,
            preferred_exact=["technology", "generation_type", "resource_type"],
            patterns=[r"technology", r"generation[_ ]type", r"resource[_ ]type", r"tech"],
        ),
        "county": _select_semantic_column(
            df,
            preferred_exact=["county", "location_county"],
            patterns=[r"county", r"location_county"],
        ),
        "capacity_mw": _select_semantic_column(
            df,
            preferred_exact=["capacity_mw", "capacity_mw_1"],
            patterns=[r"capacity.*mw", r"\bmw\b", r"size"],
            numeric_required=True,
        ),
        "cod_date": _select_semantic_column(
            df,
            preferred_exact=["projected_cod", "commercial_operation_date", "cod", "in_service"],
            patterns=[
                r"proposed.*completion",
                r"commercial.*operation",
                r"cod",
                r"in.?service",
            ],
            date_required=True,
        ),
        "developer": _select_semantic_column(
            df,
            preferred_exact=["interconnecting_entity", "developer", "owner", "entity"],
            patterns=[r"interconnecting.*entity", r"developer", r"owner", r"entity"],
        ),
        "reporting_zone": _select_semantic_column(
            df,
            preferred_exact=["cdr_reporting_zone", "reporting_zone", "zone", "region"],
            patterns=[r"reporting.*zone", r"zone", r"region"],
        ),
        "project_name": _select_semantic_column(
            df,
            preferred_exact=["project_name", "unit_name"],
            patterns=[r"project.*name", r"resource.*name", r"\bname\b"],
        ),
        "queue_id": _select_semantic_column(
            df,
            preferred_exact=["inr", "ginr", "gir", "queue_id", "project_id", "id"],
            patterns=[
                r"queue.*(id|number)",
                r"project.*id",
                r"^id$",
                r"^inr$",
                r"^ginr$",
                r"^gir$",
            ],
        ),
    }


def _select_semantic_column(
    df: pd.DataFrame,
    preferred_exact: list[str],
    patterns: list[str],
    *,
    numeric_required: bool = False,
    date_required: bool = False,
) -> str | None:
    candidates: list[tuple[int, str]] = []

    lower_to_original = {column.lower(): column for column in df.columns}
    for idx, exact in enumerate(preferred_exact):
        original = lower_to_original.get(exact.lower())
        if original:
            base_score = 200 - (idx * 10)
            total = _semantic_data_score(df, original, base_score, numeric_required, date_required)
            candidates.append((total, original))

    for column in df.columns:
        col_lower = column.lower()
        pattern_hits = 0
        for idx, pattern in enumerate(patterns):
            if re.search(pattern, col_lower):
                pattern_hits += max(1, 30 - idx)
        if pattern_hits == 0:
            continue
        total = _semantic_data_score(df, column, pattern_hits, numeric_required, date_required)
        candidates.append((total, column))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_column = candidates[0]
    if best_score <= 0:
        return None
    return best_column


def _semantic_data_score(
    df: pd.DataFrame,
    column: str,
    base_score: int,
    numeric_required: bool,
    date_required: bool,
) -> int:
    series = df[column]
    if isinstance(series, pd.DataFrame):
        return -999

    score = base_score
    if len(series):
        non_na_ratio = float(series.notna().mean())
        unique_count = int(series.nunique(dropna=True))
    else:
        non_na_ratio = 0.0
        unique_count = 0

    score += int(non_na_ratio * 80)
    score += min(unique_count, 25)

    if numeric_required:
        numeric_ratio = float(pd.to_numeric(series, errors="coerce").notna().mean()) if len(series) else 0.0
        score += int(numeric_ratio * 90)
        if numeric_ratio < 0.2:
            score -= 200

    if date_required:
        date_ratio = float(pd.to_datetime(series, errors="coerce").notna().mean()) if len(series) else 0.0
        score += int(date_ratio * 90)
        if date_ratio < 0.2:
            score -= 200

    col_lower = column.lower()
    if col_lower.startswith("unnamed"):
        score -= 150
    if "tables_that_provide" in col_lower:
        score -= 80
    if len(col_lower) > 70:
        score -= 20

    return score


def _select_key_columns(df: pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    priority_patterns = [
        r"queue.*(id|number)",
        r"project.*id",
        r"^id$",
        r"^inr$",
        r"^ginr$",
        r"^gir$",
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


def _dedupe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    all_names: set[str] = set(df.columns)

    for column in df.columns:
        count = seen.get(column, 0)
        if count == 0:
            deduped.append(column)
        else:
            # Find a suffix that doesn't clash with any existing column name
            candidate = f"{column}_{count}"
            while candidate in all_names:
                count += 1
                candidate = f"{column}_{count}"
            deduped.append(candidate)
            all_names.add(candidate)
        seen[column] = count + 1

    df.columns = deduped
    return df


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


def _remap_storage_fuel(df: pd.DataFrame) -> pd.DataFrame:
    """Remaps 'Other' fuel type to 'Battery / Storage' if keywords are found."""
    semantic = infer_semantic_columns(df)
    fuel_col = semantic.get("fuel")
    if not fuel_col or fuel_col not in df.columns:
        return df

    # Keywords that indicate storage/battery
    storage_pattern = r"(?i)battery|storage|esr|bss|ess"

    # Identify potential content columns to check for keywords
    check_cols = [
        col
        for col in df.columns
        if re.search(r"technology|generation|type|project|name", col, re.IGNORECASE)
    ]

    mask_other = df[fuel_col].astype(str).str.contains(r"(?i)^other$", na=False)

    def _is_storage(row: pd.Series) -> bool:
        combined_text = " ".join(str(row.get(col, "")) for col in check_cols)
        return bool(re.search(storage_pattern, combined_text))

    # Use a collision-safe temp column name
    _tmp_col = "__is_storage_detected__"
    df.loc[mask_other, _tmp_col] = df[mask_other].apply(_is_storage, axis=1)
    df.loc[df[_tmp_col].eq(True), fuel_col] = "Battery / Storage"

    return df.drop(columns=[_tmp_col], errors="ignore")


def _to_key_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()
