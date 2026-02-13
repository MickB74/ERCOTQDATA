from __future__ import annotations

import io
import os
import re
import subprocess
import zipfile
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

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
    """Fetches ERCOT interconnection queue data from ERCOT-hosted sources only."""
    errors: list[str] = []

    if custom_url:
        _assert_ercot_url(custom_url, "Custom URL")
        df = _download_table(custom_url)
        if df.empty:
            raise RuntimeError(f"Custom ERCOT URL returned no rows: {custom_url}")
        return df, {"source": "ercot_custom", "source_url": custom_url}

    index_urls = list(DEFAULT_REPORT_INDEX_URLS)
    env_index_url = os.environ.get("ERCOT_REPORT_INDEX_URL")
    if env_index_url:
        _assert_ercot_url(env_index_url, "ERCOT_REPORT_INDEX_URL")
        index_urls.insert(0, env_index_url)

    for index_url in index_urls:
        try:
            _assert_ercot_url(index_url, "ERCOT index URL")
            report_url = discover_latest_report_url(index_url)
            _assert_ercot_url(report_url, "Discovered report URL")

            df = _download_table(report_url)
            if not df.empty:
                return df, {
                    "source": "ercot_mis",
                    "source_url": report_url,
                    "index_url": index_url,
                }
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{index_url}: {exc}")

    error_body = " | ".join(errors) if errors else "No ERCOT source returned data."
    raise RuntimeError(
        "Could not fetch ERCOT interconnection queue data from ERCOT-hosted sources. "
        "Set a direct ERCOT file URL in the app sidebar and try again. Details: "
        f"{error_body}"
    )


def discover_latest_report_url(index_url: str) -> str:
    content, _, effective_url = _fetch_url(index_url)
    html_text = content.decode("utf-8", errors="ignore")

    candidates = _extract_report_candidates(html_text, effective_url)
    candidates = [candidate for candidate in candidates if _is_ercot_url(str(candidate["url"]))]
    if not candidates:
        raise RuntimeError("No ERCOT report links found on ERCOT index page")

    candidates.sort(key=lambda item: (item["date"], item["score"], item["position"]), reverse=True)
    return str(candidates[0]["url"])


def _fetch_url(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS) -> tuple[bytes, str, str]:
    """Fetches URL content using requests with a curl fallback for TLS edge-cases."""
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            verify=False,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        return response.content, content_type, response.url
    except Exception as request_exc:  # pylint: disable=broad-except
        curl_cmd = [
            "curl",
            "-k",
            "-L",
            "-sS",
            "--ciphers",
            "DEFAULT@SECLEVEL=1",
            "-A",
            headers["User-Agent"],
            url,
        ]
        try:
            output = subprocess.check_output(curl_cmd, timeout=timeout + 10)
            return output, "", url
        except Exception as curl_exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Request failed via requests ({request_exc}) and curl fallback ({curl_exc})"
            ) from request_exc


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
    keywords = (
        "gis",
        "interconnection",
        "queue",
        "status report",
        "doclookupid",
        "getreport",
        "mirdownload",
    )
    if any(keyword in combined for keyword in keywords):
        return True

    if label_lower in ("xlsx", "xls", "csv", "zip") and "doclookupid=" in url_lower:
        return True

    return False


def _score_candidate(url_lower: str, label_lower: str) -> int:
    score = 0
    combined = f"{url_lower} {label_lower}"

    if "gis" in combined:
        score += 5
    if "interconnection" in combined or "queue" in combined:
        score += 3
    if "status" in combined:
        score += 1
    if "doclookupid" in url_lower or "getreport" in url_lower or "mirdownload" in url_lower:
        score += 4
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

    content, content_type, effective_url = _fetch_url(url)
    url_lower = effective_url.lower()

    if _looks_like_html(content, content_type):
        html_text = content.decode("utf-8", errors="ignore")

        try:
            tables = pd.read_html(io.StringIO(html_text))
            if tables:
                largest_table = max(tables, key=len)
                if len(largest_table) > 0:
                    return largest_table
        except ValueError:
            pass

        nested_candidates = _extract_report_candidates(html_text, effective_url)
        nested_candidates = [candidate for candidate in nested_candidates if _is_ercot_url(str(candidate["url"]))]
        if nested_candidates:
            nested_candidates.sort(key=lambda item: (item["date"], item["score"], item["position"]), reverse=True)
            return _download_table(str(nested_candidates[0]["url"]), depth=depth + 1)

        raise RuntimeError(f"URL did not return tabular data: {effective_url}")

    if url_lower.endswith(".zip") or "zip" in content_type or content[:2] == b"PK":
        return _read_zip_table(content)

    return _read_table_bytes(content, effective_url, content_type)


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
        return _read_excel_with_fallbacks(content)

    if source_lower.endswith(".csv") or "csv" in content_type or "text/plain" in content_type:
        return _read_csv_with_fallbacks(content)

    try:
        return _read_excel_with_fallbacks(content)
    except Exception:  # pylint: disable=broad-except
        return _read_csv_with_fallbacks(content)


def _read_excel_with_fallbacks(content: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(content))
    candidate_tables: list[tuple[int, pd.DataFrame]] = []

    for sheet_name in xls.sheet_names:
        for header_row in (0, 1, 2, 3, 4):
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
            except Exception:  # pylint: disable=broad-except
                continue

            if df is None or df.empty:
                continue

            cleaned = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
            if cleaned.empty:
                continue

            score = _score_table_columns(cleaned.columns)
            score += min(len(cleaned), 1000) // 100
            candidate_tables.append((score, cleaned))

    if not candidate_tables:
        return pd.DataFrame()

    candidate_tables.sort(key=lambda item: item[0], reverse=True)
    return candidate_tables[0][1]


def _score_table_columns(columns: Any) -> int:
    score = 0
    column_text = " ".join(str(col).lower() for col in columns)

    if "queue" in column_text and "id" in column_text:
        score += 10
    if "project" in column_text:
        score += 6
    if "status" in column_text:
        score += 5
    if "mw" in column_text or "capacity" in column_text:
        score += 5
    if "county" in column_text:
        score += 4
    if "fuel" in column_text or "technology" in column_text:
        score += 4

    return score


def _read_csv_with_fallbacks(content: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding, low_memory=False)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError("Could not parse CSV data. " + " | ".join(errors))


def _assert_ercot_url(url: str, label: str) -> None:
    if not _is_ercot_url(url):
        raise ValueError(f"{label} must be hosted on an ERCOT domain (*.ercot.com): {url}")


def _is_ercot_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    return host.endswith("ercot.com")


def _looks_like_html(content: bytes, content_type: str) -> bool:
    if "html" in content_type:
        return True

    trimmed = content.lstrip()
    return trimmed.startswith((b"<!DOCTYPE html", b"<html"))
