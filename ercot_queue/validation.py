from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ercot_queue.config import REQUEST_TIMEOUT_SECONDS

INTERCONNECTION_FYI_ERCOT_URL = "https://www.interconnection.fyi/projects/market/ERCOT"
KNOWN_STATUS_TOKENS = [
    "Operational",
    "Withdrawn",
    "Suspended",
    "Active",
    "Queued",
    "In Service",
    "Cancelled",
    "Pending",
]
QUEUE_ID_PATTERN = re.compile(r"\b\d{2}[A-Za-z]{3}\d{4}[A-Za-z0-9]*\b")


def fetch_interconnection_fyi_ercot(url: str = INTERCONNECTION_FYI_ERCOT_URL) -> tuple[pd.DataFrame, dict[str, Any]]:
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    html = response.text

    parsers = [
        _parse_html_tables,
        _parse_from_next_data,
        _parse_from_dom_rows,
    ]

    for parser in parsers:
        parsed_df = parser(html)
        canonical_df = _to_canonical_external(parsed_df)
        if not canonical_df.empty:
            metadata = {
                "source": "interconnection_fyi",
                "source_url": url,
                "parser": parser.__name__,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            return canonical_df, metadata

    raise RuntimeError("Could not parse ERCOT projects from Interconnection.fyi page")


def compare_local_to_external(
    local_df: pd.DataFrame,
    external_df: pd.DataFrame,
    *,
    local_queue_col: str | None,
    local_status_col: str | None,
    local_capacity_col: str | None,
    tolerance_mw: float = 1.0,
) -> dict[str, Any]:
    local_cmp = _prepare_local_comparison(local_df, local_queue_col, local_status_col, local_capacity_col)
    external_cmp = _prepare_external_comparison(external_df)

    local_keys = set(local_cmp["queue_id_norm"].dropna())
    external_keys = set(external_cmp["queue_id_norm"].dropna())
    matched_keys = local_keys & external_keys

    missing_in_local_keys = sorted(external_keys - local_keys)
    missing_in_external_keys = sorted(local_keys - external_keys)

    local_map = local_cmp.set_index("queue_id_norm", drop=False)
    external_map = external_cmp.set_index("queue_id_norm", drop=False)

    status_mismatches: list[dict[str, Any]] = []
    capacity_mismatches: list[dict[str, Any]] = []

    for key in sorted(matched_keys):
        local_row = local_map.loc[key]
        external_row = external_map.loc[key]

        local_status = _normalize_status(local_row.get("status"))
        external_status = _normalize_status(external_row.get("status"))
        if local_status and external_status and local_status != external_status:
            status_mismatches.append(
                {
                    "queue_id": key,
                    "local_status": local_row.get("status"),
                    "external_status": external_row.get("status"),
                    "project_name": local_row.get("project_name") or external_row.get("project_name"),
                }
            )

        local_capacity = pd.to_numeric(pd.Series([local_row.get("capacity_mw")]), errors="coerce").iloc[0]
        external_capacity = pd.to_numeric(pd.Series([external_row.get("capacity_mw")]), errors="coerce").iloc[0]
        if pd.notna(local_capacity) and pd.notna(external_capacity):
            delta = float(local_capacity - external_capacity)
            if abs(delta) > tolerance_mw:
                capacity_mismatches.append(
                    {
                        "queue_id": key,
                        "project_name": local_row.get("project_name") or external_row.get("project_name"),
                        "local_status": local_row.get("status"),
                        "external_status": external_row.get("status"),
                        "local_capacity_mw": float(local_capacity),
                        "external_capacity_mw": float(external_capacity),
                        "delta_mw": delta,
                    }
                )

    missing_in_local = external_cmp[external_cmp["queue_id_norm"].isin(missing_in_local_keys)].copy()
    missing_in_external = local_cmp[local_cmp["queue_id_norm"].isin(missing_in_external_keys)].copy()

    # Calculate MW sums for summary
    local_mw = float(local_cmp["capacity_mw"].sum())
    external_mw = float(external_cmp["capacity_mw"].sum())
    matched_mw = float(local_cmp[local_cmp["queue_id_norm"].isin(matched_keys)]["capacity_mw"].sum())
    missing_in_local_mw = float(missing_in_local["capacity_mw"].sum())
    missing_in_external_mw = float(missing_in_external["capacity_mw"].sum())
    status_mismatch_mw = float(local_cmp[local_cmp["queue_id_norm"].isin([m["queue_id"] for m in status_mismatches])]["capacity_mw"].sum())
    capacity_mismatch_mw = float(local_cmp[local_cmp["queue_id_norm"].isin([m["queue_id"] for m in capacity_mismatches])]["capacity_mw"].sum())

    return {
        "summary": {
            "local_queue_ids": int(len(local_keys)),
            "local_mw": local_mw,
            "external_queue_ids": int(len(external_keys)),
            "external_mw": external_mw,
            "matched_queue_ids": int(len(matched_keys)),
            "matched_mw": matched_mw,
            "missing_in_local": int(len(missing_in_local_keys)),
            "missing_in_local_mw": missing_in_local_mw,
            "missing_in_external": int(len(missing_in_external_keys)),
            "missing_in_external_mw": missing_in_external_mw,
            "status_mismatches": int(len(status_mismatches)),
            "status_mismatch_mw": status_mismatch_mw,
            "capacity_mismatches": int(len(capacity_mismatches)),
            "capacity_mismatch_mw": capacity_mismatch_mw,
        },
        "missing_in_local": missing_in_local,
        "missing_in_external": missing_in_external,
        "status_mismatches": pd.DataFrame(status_mismatches),
        "capacity_mismatches": pd.DataFrame(capacity_mismatches),
    }


def _parse_html_tables(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame()

    for table in tables:
        cols = {str(col).strip().lower() for col in table.columns}
        if any("queue" in col and "id" in col for col in cols):
            return table

    return pd.DataFrame()


def _parse_from_next_data(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        return pd.DataFrame()

    try:
        payload = json.loads(script.string)
    except json.JSONDecodeError:
        return pd.DataFrame()

    candidate_lists: list[list[dict[str, Any]]] = []

    # Iterative traversal to avoid hitting Python's recursion limit on deep JSON trees
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            if node and all(isinstance(item, dict) for item in node):
                score = sum(1 for item in node if _looks_like_project_dict(item))
                if score >= max(3, int(0.4 * len(node))):
                    candidate_lists.append(node)
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())

    if not candidate_lists:
        return pd.DataFrame()

    best = max(candidate_lists, key=len)
    return pd.DataFrame(best)


def _parse_from_dom_rows(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if "/project/" not in href:
            continue

        project_name = " ".join(anchor.get_text(" ", strip=True).split())
        if not project_name:
            continue

        row_text = ""
        node = anchor
        for _ in range(7):
            row_text = " ".join(node.get_text(" ", strip=True).split())
            if QUEUE_ID_PATTERN.search(row_text):
                break
            if node.parent is None:
                break
            node = node.parent

        parsed = _parse_project_row_text(row_text, project_name)
        queue_id = parsed.get("queue_id")
        if not queue_id or queue_id in seen_ids:
            continue

        rows.append(parsed)
        seen_ids.add(queue_id)

    return pd.DataFrame(rows)


def _parse_project_row_text(row_text: str, project_name: str) -> dict[str, Any]:
    queue_match = QUEUE_ID_PATTERN.search(row_text)
    if not queue_match:
        return {"project_name": project_name}

    queue_id = queue_match.group(0)
    after_queue = row_text[queue_match.end() :].strip()

    state = None
    county = None
    status = None

    state_match = re.match(r"^(?P<state>[A-Z]{2})\s+(?P<rest>.+)$", after_queue)
    if state_match:
        state = state_match.group("state")
        rest = state_match.group("rest").strip()
    else:
        rest = after_queue

    for token in sorted(KNOWN_STATUS_TOKENS, key=len, reverse=True):
        if rest.lower().endswith(token.lower()):
            status = token
            county = rest[: -len(token)].strip() or None
            break

    if status is None:
        county = rest or None

    return {
        "project_name": project_name,
        "queue_id": queue_id,
        "state": state,
        "county": county,
        "status": status,
    }


def _to_canonical_external(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "project_name",
                "queue_id",
                "state",
                "county",
                "status",
                "capacity_mw",
            ]
        )

    normalized = df.copy()
    normalized.columns = [_normalize_col_name(col) for col in normalized.columns]

    col_map = {
        "project_name": _first_matching(normalized.columns, [r"project.*name", r"name"]),
        "queue_id": _first_matching(normalized.columns, [r"queue.*id", r"project.*id", r"^id$"]),
        "state": _first_matching(normalized.columns, [r"state"]),
        "county": _first_matching(normalized.columns, [r"county"]),
        "status": _first_matching(normalized.columns, [r"status"]),
        "capacity_mw": _first_matching(normalized.columns, [r"capacity.*mw", r"mw", r"size"]),
    }

    canonical = pd.DataFrame()
    for canonical_col, source_col in col_map.items():
        canonical[canonical_col] = normalized[source_col] if source_col else pd.NA

    canonical["queue_id"] = canonical["queue_id"].astype("string").str.strip()
    canonical = canonical[canonical["queue_id"].notna()]

    canonical["capacity_mw"] = pd.to_numeric(canonical["capacity_mw"], errors="coerce")
    canonical = canonical.drop_duplicates(subset=["queue_id"], keep="last")

    return canonical.reset_index(drop=True)


def _looks_like_project_dict(item: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in item.keys()}
    return any("queue" in key and "id" in key for key in keys) and any("status" in key for key in keys)


def _prepare_local_comparison(
    local_df: pd.DataFrame,
    local_queue_col: str | None,
    local_status_col: str | None,
    local_capacity_col: str | None,
) -> pd.DataFrame:
    if not local_queue_col or local_queue_col not in local_df.columns:
        return pd.DataFrame(columns=["queue_id_norm", "queue_id", "project_name", "status", "capacity_mw"])

    local = pd.DataFrame()
    local["queue_id"] = local_df[local_queue_col]
    local["queue_id_norm"] = local["queue_id"].map(_normalize_queue_id)

    if local_status_col and local_status_col in local_df.columns:
        local["status"] = local_df[local_status_col]
    else:
        local["status"] = pd.NA

    if local_capacity_col and local_capacity_col in local_df.columns:
        local["capacity_mw"] = pd.to_numeric(local_df[local_capacity_col], errors="coerce")
    else:
        local["capacity_mw"] = pd.NA

    project_col = _first_matching(local_df.columns, [r"project.*name", r"resource.*name", r"name"])
    local["project_name"] = local_df[project_col] if project_col else pd.NA

    local = local[local["queue_id_norm"].notna()]
    local = local.drop_duplicates(subset=["queue_id_norm"], keep="last")
    return local.reset_index(drop=True)


def _prepare_external_comparison(external_df: pd.DataFrame) -> pd.DataFrame:
    external = external_df.copy()
    external["queue_id_norm"] = external["queue_id"].map(_normalize_queue_id)
    external["capacity_mw"] = pd.to_numeric(external["capacity_mw"], errors="coerce")
    external = external[external["queue_id_norm"].notna()]
    external = external.drop_duplicates(subset=["queue_id_norm"], keep="last")
    return external.reset_index(drop=True)


def _normalize_queue_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", str(value).upper())
    return text or None


def _normalize_status(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).lower().split())
    return text or None


def _first_matching(columns: list[str], patterns: list[str]) -> str | None:
    for pattern in patterns:
        regex = re.compile(pattern)
        for column in columns:
            if regex.search(column):
                return column
    return None


def _normalize_col_name(column: Any) -> str:
    text = str(column).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text
