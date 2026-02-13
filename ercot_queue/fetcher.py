from __future__ import annotations

import io
import os
import re
import zipfile
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ercot_queue.config import DEFAULT_REPORT_INDEX_URLS, REQUEST_TIMEOUT_SECONDS

DATA_EXTENSIONS = (".csv", ".xls", ".xlsx", ".zip")
DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})"),
    re.compile(r"(\d{1,2})[-_/](\d{1,2})[-_/](20\d{2})"),
]


def fetch_latest_ercot_queue(custom_url: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetches ERCOT interconnection queue data from the best available source."""
    errors: list[str] = []

    if custom_url:
        df = _download_table(custom_url)
        return df, {"source": "custom_url", "source_url": custom_url}

    gridstatus_df = _try_gridstatus(errors)
    if gridstatus_df is not None and not gridstatus_df.empty:
        return gridstatus_df, {"source": "gridstatus", "source_url": "gridstatus.Ercot().get_interconnection_queue"}

    index_urls = list(DEFAULT_REPORT_INDEX_URLS)
    env_index_url = os.environ.get("ERCOT_REPORT_INDEX_URL")
    if env_index_url:
        index_urls.insert(0, env_index_url)

    for index_url in index_urls:
        try:
            report_url = discover_latest_report_url(index_url)
            df = _download_table(report_url)
            if not df.empty:
                return df, {
                    "source": "ercot_mis",
                    "source_url": report_url,
                    "index_url": index_url,
                }
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{index_url}: {exc}")

    error_body = " | ".join(errors) if errors else "No source returned data."
    raise RuntimeError(
        "Could not fetch ERCOT interconnection queue data. "
        "Set a direct URL in the app sidebar and try again. Details: "
        f"{error_body}"
    )


def _try_gridstatus(errors: list[str]) -> pd.DataFrame | None:
    try:
        import gridstatus  # type: ignore

        iso = gridstatus.Ercot()
        return iso.get_interconnection_queue()
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"gridstatus: {exc}")
        return None


def discover_latest_report_url(index_url: str) -> str:
    response = requests.get(index_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    candidates = _extract_report_candidates(response.text, response.url)
    if not candidates:
        raise RuntimeError("No report links found on ERCOT index page")

    candidates.sort(key=lambda item: (item["date"], item["score"], item["position"]), reverse=True)
    return str(candidates[0]["url"])


def _extract_report_candidates(html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, anchor in enumerate(soup.find_all("a")):
        href = (anchor.get("href") or "").strip()
        label = " ".join(anchor.get_text(" ", strip=True).split())
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue

        abs_url = urljoin(base_url, href)
        normalized = abs_url.lower()
        label_lower = label.lower()

        if normalized in seen:
            continue

        if not _looks_like_report_link(normalized, label_lower):
            continue

        score = _score_candidate(normalized, label_lower)
        found_date = _parse_date(normalized) or _parse_date(label)
        results.append(
            {
                "url": abs_url,
                "label": label,
                "score": score,
                "date": found_date or datetime.min,
                "position": idx,
            }
        )
        seen.add(normalized)

    return results


def _looks_like_report_link(url_lower: str, label_lower: str) -> bool:
    if any(url_lower.endswith(ext) for ext in DATA_EXTENSIONS):
        return True

    combined = f"{url_lower} {label_lower}"
    keywords = ("gis", "interconnection", "queue", "status report", "doclookupid", "getreport")
    return any(keyword in combined for keyword in keywords)


def _score_candidate(url_lower: str, label_lower: str) -> int:
    score = 0
    combined = f"{url_lower} {label_lower}"

    if "gis" in combined:
        score += 3
    if "interconnection" in combined or "queue" in combined:
        score += 3
    if "status" in combined:
        score += 1
    if "doclookupid" in combined or "getreport" in combined:
        score += 2
    if any(url_lower.endswith(ext) for ext in DATA_EXTENSIONS):
        score += 1

    return score


def _parse_date(value: str) -> datetime | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(value)
        if not match:
            continue

        parts = [int(part) for part in match.groups()]
        if parts[0] > 1900:
            year, month, day = parts
        else:
            month, day, year = parts

        try:
            return datetime(year, month, day)
        except ValueError:
            continue

    return None


def _download_table(url: str, depth: int = 0) -> pd.DataFrame:
    if depth > 2:
        raise RuntimeError("Exceeded redirect/discovery depth while fetching report data")

    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    content = response.content
    content_type = response.headers.get("content-type", "").lower()
    url_lower = url.lower()

    if "html" in content_type or content.lstrip().startswith((b"<!DOCTYPE html", b"<html")):
        html_text = response.text

        try:
            tables = pd.read_html(io.StringIO(html_text))
            if tables:
                largest_table = max(tables, key=len)
                if len(largest_table) > 0:
                    return largest_table
        except ValueError:
            pass

        nested_candidates = _extract_report_candidates(html_text, response.url)
        if nested_candidates:
            nested_candidates.sort(key=lambda item: (item["date"], item["score"], item["position"]), reverse=True)
            return _download_table(str(nested_candidates[0]["url"]), depth=depth + 1)

        raise RuntimeError(f"URL did not return file data: {url}")

    if url_lower.endswith(".zip") or "zip" in content_type or content[:2] == b"PK":
        return _read_zip_table(content)

    return _read_table_bytes(content, url, content_type)


def _read_zip_table(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = sorted(zf.namelist())
        for name in names:
            name_lower = name.lower()
            if not any(name_lower.endswith(ext) for ext in (".csv", ".xls", ".xlsx")):
                continue
            with zf.open(name) as file_handle:
                return _read_table_bytes(file_handle.read(), name, "")

    raise RuntimeError("ZIP file did not contain CSV/XLS/XLSX data")


def _read_table_bytes(content: bytes, source_name: str, content_type: str) -> pd.DataFrame:
    source_lower = source_name.lower()

    if source_lower.endswith((".xls", ".xlsx")) or "excel" in content_type or "spreadsheet" in content_type:
        try:
            xls_data = io.BytesIO(content)
            xls = pd.ExcelFile(xls_data)
            sheet_dfs = []
            valid_sheets = []
            
            for sheet_name in xls.sheet_names:
                # Read the first 10 rows to find the header
                sample_df = pd.read_excel(xls, sheet_name=sheet_name, nrows=10, header=None)
                if sample_df.empty:
                    continue
                
                header_row_index = -1
                for i, row in sample_df.iterrows():
                    row_joined = " ".join(str(val).lower() for val in row.dropna())
                    if any(term in row_joined for term in ["queue", "project", "gir", "inr", "record"]):
                        header_row_index = i
                        break
                
                if header_row_index != -1:
                    # Found a potential header row, re-read from there
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row_index)
                else:
                    # Try default read if no specific row found
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                
                if df.empty or len(df) < 1:
                    continue
                
                # Double check columns for sanity
                cols = [str(c).lower() for c in df.columns]
                is_valid = any(
                    any(term in col for term in ["queue", "project", "gir", "inr"])
                    for col in cols
                )
                
                if is_valid:
                    valid_sheets.append(sheet_name)
                    # Clean up columns - sometimes excel adds 'Unnamed' columns
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
                    sheet_dfs.append(df)
            
            if not sheet_dfs:
                return pd.DataFrame()
            
            if len(sheet_dfs) == 1:
                return sheet_dfs[0]
                
            # Aggregate sheets
            combined = pd.concat(sheet_dfs, ignore_index=True, sort=False)
            return combined
            
        except Exception as exc:
            # Fallback to single read if complex read fails
            try:
                xls_data.seek(0)
                return pd.read_excel(xls_data)
            except:
                pass

    if source_lower.endswith(".csv") or "csv" in content_type or "text/plain" in content_type:
        return _read_csv_with_fallbacks(content)

    try:
        return pd.read_excel(io.BytesIO(content))
    except Exception:  # pylint: disable=broad-except
        return _read_csv_with_fallbacks(content)


def _read_csv_with_fallbacks(content: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding, low_memory=False)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError("Could not parse CSV data. " + " | ".join(errors))
