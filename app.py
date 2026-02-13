import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

# Standardize ercot_queue package access
try:
    from ercot_queue.config import DEFAULT_DATA_PRODUCT_URLS, MAX_CHANGE_SAMPLE_ROWS
    from ercot_queue.diffing import calculate_diff
    from ercot_queue.fetcher import (
        discover_latest_report_url,
        discover_report_index_from_product_page,
        fetch_latest_ercot_queue,
        fetch_summary_under_study_mw,
    )
    from ercot_queue.processing import infer_semantic_columns, prepare_queue_dataframe
except ImportError as exc:
    st.error(f"Critical Package Error: {exc}")
    st.stop()
from ercot_queue.store import (
    ensure_data_dirs,
    load_change_report,
    load_latest_snapshot,
    load_snapshot_history,
    save_snapshot,
)

FUEL_CODE_MAP = {
    "BIO": "Biomass",
    "COA": "Coal",
    "GAS": "Gas",
    "GEO": "Geothermal",
    "HYD": "Hydrogen",
    "MWH": "Miscellaneous Wind Hybrid",
    "NUC": "Nuclear",
    "OIL": "Fuel oil",
    "OTH": "Other",
    "PET": "Petcoke",
    "SOL": "Solar",
    "WAT": "Water",
    "WIN": "Wind",
}

TECHNOLOGY_CODE_MAP = {
    "BA": "Battery Energy Storage",
    "CC": "Combined-cycle",
    "CE": "Compressed air energy storage",
    "CP": "Concentrated solar power",
    "EN": "Energy storage",
    "FC": "Fuel cell",
    "GT": "Combustion turbine (simple-cycle)",
    "HY": "Hydroelectric turbine",
    "IC": "Internal combustion engine (reciprocating)",
    "OT": "Other",
    "PV": "Photovoltaic solar",
    "ST": "Steam turbine (non-CC)",
    "WT": "Wind turbine",
}


def _format_timestamp(raw_ts: str) -> str:
    try:
        parsed = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return raw_ts


@st.cache_data(ttl=60 * 30, show_spinner=False)
def _cached_fetch_under_study_capacity(source_url: str) -> float | None:
    return fetch_summary_under_study_mw(source_url)


def _map_code_to_name(value: object, crosswalk: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return crosswalk.get(text.upper(), text)


def _apply_crosswalk_columns(df: pd.DataFrame, semantic: dict[str, str | None]) -> pd.DataFrame:
    result = df.copy()

    fuel_col = semantic.get("fuel")
    if fuel_col and fuel_col in result.columns:
        result["fuel_name"] = result[fuel_col].map(lambda value: _map_code_to_name(value, FUEL_CODE_MAP))

    technology_col = semantic.get("technology")
    if technology_col and technology_col in result.columns:
        result["technology_name"] = result[technology_col].map(
            lambda value: _map_code_to_name(value, TECHNOLOGY_CODE_MAP)
        )

    return result


def _extract_under_study_capacity_mw(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None

    text_cols = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])
    ]
    if not text_cols:
        return None

    text_frame = df[text_cols].astype("string")
    match_mask = text_frame.apply(
        lambda col: col.str.contains("total capacity under study", case=False, na=False, regex=False)
    ).any(axis=1)
    if not match_mask.any():
        return None

    matched_rows = df[match_mask]
    numeric_rows = matched_rows.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    values = [float(value) for value in numeric_rows.to_numpy().ravel() if pd.notna(value) and float(value) > 0]
    if not values:
        return None

    return max(values)


def _extract_project_detail_capacity_mw(df: pd.DataFrame, capacity_column: str | None) -> float | None:
    if df.empty or not capacity_column or capacity_column not in df.columns or "source_sheet" not in df.columns:
        return None

    source_text = df["source_sheet"].astype("string").str.lower()
    project_mask = source_text.str.contains("project details - large gen|project details - small gen", na=False)
    project_rows = df[project_mask]
    if project_rows.empty:
        return None

    capacity = pd.to_numeric(project_rows[capacity_column], errors="coerce")
    if not capacity.notna().any():
        return None

    return float(capacity.sum(skipna=True))


def _normalize_filter_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()
    return text if text else "Unknown"


def _extract_plotly_points(event: object) -> list[Any]:
    if event is None:
        return []

    if isinstance(event, dict):
        selection = event.get("selection")
        if isinstance(selection, dict):
            points = selection.get("points")
            if isinstance(points, list):
                return points
        return []

    selection = getattr(event, "selection", None)
    if selection is None:
        return []

    points = getattr(selection, "points", None)
    if isinstance(points, list):
        return points
    return []


def _selection_point_value(point: Any, field: str) -> object:
    if isinstance(point, dict):
        return point.get(field)
    return getattr(point, field, None)


def _selected_values_from_event(event: object, field: str) -> list[str]:
    values: set[str] = set()
    for point in _extract_plotly_points(event):
        raw = _selection_point_value(point, field)
        if raw is None:
            continue
        values.add(_normalize_filter_value(raw))
    return sorted(values)


def _selected_customdata_values(event: object, index: int) -> list[str]:
    values: set[str] = set()
    for point in _extract_plotly_points(event):
        custom_data = _selection_point_value(point, "customdata")
        if not isinstance(custom_data, (list, tuple)):
            continue
        if len(custom_data) <= index:
            continue
        values.add(_normalize_filter_value(custom_data[index]))
    return sorted(values)


def _update_chart_filter(filter_key: str, values: list[str]) -> bool:
    if not values:
        return False

    chart_filters = st.session_state.setdefault("chart_filters", {})
    existing = chart_filters.get(filter_key, [])
    selection_mode = st.session_state.get("chart_selection_mode", "Add to selection")

    if selection_mode == "Add to selection":
        merged_values = sorted(set(existing).union(values))
    else:
        merged_values = values

    if existing == merged_values:
        return False

    chart_filters[filter_key] = merged_values
    return True


def _style_chart(fig: Any, *, x_tick_angle: int = -30, height: int = 380) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(tickangle=x_tick_angle, automargin=True)
    fig.update_yaxes(automargin=True)


def _apply_chart_filters(
    df: pd.DataFrame,
    chart_filters: dict[str, list[str]],
    *,
    fuel_display_col: str | None,
    technology_display_col: str | None,
    developer_col: str | None,
    zone_col: str | None,
    cod_col: str | None,
    exclude_key: str | None = None,
) -> pd.DataFrame:
    filtered = df

    if exclude_key != "fuel" and chart_filters.get("fuel") and fuel_display_col and fuel_display_col in filtered.columns:
        selected_fuel = set(chart_filters["fuel"])
        filtered = filtered[filtered[fuel_display_col].map(_normalize_filter_value).isin(selected_fuel)]

    if (
        exclude_key != "technology"
        and chart_filters.get("technology")
        and technology_display_col
        and technology_display_col in filtered.columns
    ):
        selected_technology = set(chart_filters["technology"])
        filtered = filtered[filtered[technology_display_col].map(_normalize_filter_value).isin(selected_technology)]

    if exclude_key != "developer" and chart_filters.get("developer") and developer_col and developer_col in filtered.columns:
        selected_developer = set(chart_filters["developer"])
        filtered = filtered[filtered[developer_col].map(_normalize_filter_value).isin(selected_developer)]

    if exclude_key != "zone" and chart_filters.get("zone") and zone_col and zone_col in filtered.columns:
        selected_zone = set(chart_filters["zone"])
        filtered = filtered[filtered[zone_col].map(_normalize_filter_value).isin(selected_zone)]

    if exclude_key != "cod_year" and chart_filters.get("cod_year") and cod_col and cod_col in filtered.columns:
        selected_years = set(chart_filters["cod_year"])
        cod_year = pd.to_datetime(filtered[cod_col], errors="coerce").dt.year
        cod_year_text = cod_year.map(lambda value: str(int(value)) if pd.notna(value) else "Unknown")
        filtered = filtered[cod_year_text.isin(selected_years)]

    return filtered


def _apply_selection_highlight(
    plot_df: pd.DataFrame,
    dimension_col: str,
    selected_values: list[str],
) -> tuple[pd.DataFrame, str | None]:
    if not selected_values or dimension_col not in plot_df.columns:
        return plot_df, None

    highlighted = plot_df.copy()
    selected_set = set(selected_values)
    highlighted["selection_state"] = highlighted[dimension_col].map(
        lambda value: "Selected" if _normalize_filter_value(value) in selected_set else "Other"
    )
    return highlighted, "selection_state"


def _sync_sidebar_filter_state_from_chart_filters(chart_filters: dict[str, list[str]]) -> None:
    chart_to_sidebar_key = {
        "fuel": "fuel",
        "technology": "technology",
        "developer": "developer",
        "zone": "reporting_zone",
    }
    for chart_key, sidebar_key in chart_to_sidebar_key.items():
        selected = chart_filters.get(chart_key)
        if not selected:
            continue
        st.session_state[f"all_{sidebar_key}"] = False
        st.session_state[f"select_{sidebar_key}"] = selected


def _source_identity(source_url: str | None) -> str:
    if not source_url:
        return ""

    parsed = urlparse(str(source_url))
    params = parse_qs(parsed.query)
    for key, values in params.items():
        if key.lower() == "doclookupid" and values:
            return f"doclookupid:{values[0]}"

    normalized_path = parsed.path.rstrip("/")
    normalized_query = parsed.query.strip()
    return f"{normalized_path}?{normalized_query}".lower()


def _resolve_latest_source_url(user_url: str) -> str:
    text = user_url.strip()
    if not text:
        raise ValueError("Missing source URL")

    if "data-product-details" in text:
        index_url, _ = discover_report_index_from_product_page(text)
        return discover_latest_report_url(index_url)

    if "GetReports.do" in text and "reportTypeId" in text:
        return discover_latest_report_url(text)

    return text


st.set_page_config(
    page_title="ERCOT Interconnection Queue",
    layout="wide",
)

ensure_data_dirs()

st.title("ERCOT Interconnection Queue Explorer")
st.caption(
    "Pull the latest queue data, filter it, visualize it, and track exactly what changed on every refresh."
)
st.markdown(
    """
<style>
  .block-container {padding-top: 1.3rem; padding-bottom: 2.2rem;}
  [data-testid="stMetric"] {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
  }
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Data Refresh")
    default_source_url = st.session_state.get(
        "ercot_source_url",
        DEFAULT_DATA_PRODUCT_URLS[0] if DEFAULT_DATA_PRODUCT_URLS else "",
    )
    custom_url = st.text_input(
        "ERCOT Source URL",
        value=default_source_url,
        help=(
            "Defaults to ERCOT data product PG7-200-ER. "
            "You can paste an ERCOT data-product-details URL or direct ERCOT file URL."
        ),
    ).strip()
    st.session_state["ercot_source_url"] = custom_url
    st.caption("Primary source: https://www.ercot.com/mp/data-products/data-product-details?id=pg7-200-er")

    if st.button("Refresh Data", type="primary", use_container_width=True):
        with st.spinner("Checking for a new ERCOT spreadsheet..."):
            previous_df, previous_meta = load_latest_snapshot()
            previous_source_url = (previous_meta or {}).get("source_url")
            expected_latest_url: str | None = None
            source_to_check = custom_url or default_source_url

            if source_to_check:
                try:
                    expected_latest_url = _resolve_latest_source_url(source_to_check)
                except Exception:
                    expected_latest_url = None

            already_current = (
                previous_meta is not None
                and expected_latest_url is not None
                and _source_identity(expected_latest_url) == _source_identity(previous_source_url)
            )

            if already_current:
                st.session_state["refresh_notice"] = (
                    "No new ERCOT spreadsheet was found. "
                    f"Using existing snapshot {previous_meta.get('snapshot_id', 'n/a')}."
                )
            else:
                latest_raw_df, source_meta = fetch_latest_ercot_queue(custom_url or None)
                latest_source_url = source_meta.get("source_url")
                same_file_after_fetch = (
                    previous_meta is not None
                    and _source_identity(str(latest_source_url or "")) == _source_identity(previous_source_url)
                )

                if same_file_after_fetch:
                    st.session_state["refresh_notice"] = (
                        "No new ERCOT spreadsheet was found. "
                        f"Using existing snapshot {previous_meta.get('snapshot_id', 'n/a')}."
                    )
                else:
                    latest_prepared_df = prepare_queue_dataframe(latest_raw_df)
                    diff_report = calculate_diff(
                        previous_df,
                        latest_prepared_df,
                        max_sample_rows=MAX_CHANGE_SAMPLE_ROWS,
                    )
                    latest_meta = save_snapshot(
                        latest_prepared_df,
                        source_metadata=source_meta,
                        diff_report=diff_report,
                    )

                    st.session_state["last_refresh"] = {
                        "snapshot_id": latest_meta["snapshot_id"],
                        "summary": diff_report.get("summary", {}),
                    }
                    st.session_state["refresh_notice"] = (
                        "Refresh complete. "
                        f"Added {diff_report['summary']['added']}, "
                        f"Removed {diff_report['summary']['removed']}, "
                        f"Changed {diff_report['summary']['changed']}."
                    )

        notice = st.session_state.get("refresh_notice")
        if notice:
            if notice.startswith("Refresh complete"):
                st.success(notice)
            else:
                st.info(notice)


current_df, current_meta = load_latest_snapshot()
if current_df is None or current_df.empty:
    st.info("No snapshot yet. Click `Refresh Data` in the sidebar to pull the first dataset.")
    st.stop()

snapshot_df = current_df.copy()
snapshot_semantic = infer_semantic_columns(snapshot_df)
official_under_study_mw = _extract_under_study_capacity_mw(snapshot_df)
if official_under_study_mw is None and current_meta:
    source_url = current_meta.get("source_url")
    if source_url:
        try:
            official_under_study_mw = _cached_fetch_under_study_capacity(str(source_url))
        except Exception:
            official_under_study_mw = None
detail_project_capacity_mw = _extract_project_detail_capacity_mw(
    snapshot_df,
    snapshot_semantic.get("capacity_mw"),
)

scope_options = [
    "Combined (Large + Small)",
    "Large Gen Only",
    "Small Gen Only",
]
source_sheet_col = "source_sheet" if "source_sheet" in current_df.columns else None
scope_notice: str | None = None

with st.sidebar:
    st.header("Dataset Scope")
    selected_scope = st.radio(
        "Project Set",
        options=scope_options,
        index=0,
        key="dataset_scope",
    )

if source_sheet_col:
    source_text = current_df[source_sheet_col].astype("string").str.lower()
    large_mask = source_text.str.contains("project details - large gen", na=False)
    small_mask = source_text.str.contains("project details - small gen", na=False)

    if selected_scope == "Large Gen Only":
        if large_mask.any():
            current_df = current_df[large_mask].copy()
        else:
            scope_notice = "Large Gen tab not found in this snapshot; showing combined data."
    elif selected_scope == "Small Gen Only":
        if small_mask.any():
            current_df = current_df[small_mask].copy()
        else:
            scope_notice = "Small Gen tab not found in this snapshot; showing combined data."
else:
    scope_notice = "Source sheet labels are unavailable in this snapshot; showing combined data."

semantic = infer_semantic_columns(current_df)
current_df = _apply_crosswalk_columns(current_df, semantic)
filtered_df = current_df.copy()
active_filters: list[str] = []

if selected_scope != "Combined (Large + Small)":
    active_filters.append(selected_scope)

capacity_col = semantic.get("capacity_mw")
status_col = semantic.get("status")
fuel_col = semantic.get("fuel")
technology_col = semantic.get("technology")
zone_col = semantic.get("reporting_zone")
developer_col = semantic.get("developer")
county_col = semantic.get("county")
cod_col = semantic.get("cod_date")
fuel_display_col = "fuel_name" if "fuel_name" in current_df.columns else fuel_col
technology_display_col = "technology_name" if "technology_name" in current_df.columns else technology_col
chart_filters: dict[str, list[str]] = st.session_state.setdefault("chart_filters", {})
# Status chart was removed; drop any stale status chart selection.
if "status" in chart_filters:
    chart_filters.pop("status", None)
_sync_sidebar_filter_state_from_chart_filters(chart_filters)

with st.sidebar:
    st.header("Filters")

    chart_key_by_sidebar_filter = {
        "fuel": "fuel",
        "technology": "technology",
        "developer": "developer",
        "reporting_zone": "zone",
    }

    filter_definitions = [
        ("Status", "Statuses", status_col, "status"),
        ("Fuel", "Fuels", fuel_display_col, "fuel"),
        ("Technology", "Technologies", technology_display_col, "technology"),
        ("Reporting Zone", "Reporting Zones", zone_col, "reporting_zone"),
        ("Developer", "Developers", developer_col, "developer"),
        ("County", "Counties", county_col, "county"),
    ]

    for label, plural_label, column, filter_key in filter_definitions:
        if not column or column not in filtered_df.columns:
            continue

        if filter_key == "developer" and capacity_col and capacity_col in current_df.columns:
            # Sort developers by total MW (descending)
            options = (
                current_df.groupby(column)[capacity_col]
                .sum()
                .sort_values(ascending=False)
                .index.dropna()
                .astype(str)
                .tolist()
            )
        else:
            options = sorted(
                {
                    str(value)
                    for value in filtered_df[column].dropna().astype(str)
                    if str(value).strip()
                }
            )

        if not options:
            continue

        select_key = f"select_{filter_key}"
        existing_selected = st.session_state.get(select_key)
        if isinstance(existing_selected, list):
            st.session_state[select_key] = [value for value in existing_selected if value in options]

        # Add "Select All" logic
        select_all = st.checkbox(f"All {plural_label}", value=True, key=f"all_{filter_key}")
        chart_key = chart_key_by_sidebar_filter.get(filter_key)
        is_chart_dimension = chart_key is not None
        if not select_all:
            selected = st.multiselect(
                f"Select {plural_label}",
                options=options,
                default=options,
                key=select_key,
            )
            if selected and set(selected) != set(options):
                if not is_chart_dimension:
                    filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected)]
                active_filters.append(label)
            elif not selected:
                # If nothing selected and not "Select All", show nothing
                filtered_df = filtered_df.iloc[0:0]
                active_filters.append(label)
            else:
                # "All" effectively selected, so no filter is applied.
                pass
            if chart_key:
                if selected and set(selected) != set(options):
                    chart_filters[chart_key] = sorted(set(selected))
                else:
                    chart_filters.pop(chart_key, None)
        else:
            if chart_key:
                chart_filters.pop(chart_key, None)

    if capacity_col and capacity_col in filtered_df.columns:
        numeric_capacity = pd.to_numeric(filtered_df[capacity_col], errors="coerce")
        if numeric_capacity.notna().any():
            minimum = float(numeric_capacity.min())
            maximum = float(numeric_capacity.max())
            if minimum < maximum:
                low, high = st.slider(
                    "Capacity (MW)",
                    min_value=minimum,
                    max_value=maximum,
                    value=(minimum, maximum),
                )
                if low > minimum or high < maximum:
                    filtered_df = filtered_df[numeric_capacity.between(low, high, inclusive="both")]
                    active_filters.append("Capacity (MW)")
            else:
                st.caption(f"Capacity: {minimum:,.0f} MW")

    if cod_col and cod_col in filtered_df.columns:
        cod_series = pd.to_datetime(filtered_df[cod_col], errors="coerce")
        if cod_series.notna().any():
            min_date = cod_series.min().date()
            max_date = cod_series.max().date()
            if min_date < max_date:
                selected_dates = st.date_input("COD / In-Service Date", (min_date, max_date))
                if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    if start_date > min_date or end_date < max_date:
                        mask = cod_series.between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
                        filtered_df = filtered_df[mask]
                        active_filters.append("COD / In-Service Date")
            else:
                st.caption(f"COD Date: {min_date}")

with st.sidebar:
    st.markdown("---")
    st.radio(
        "Chart Selection Mode",
        options=["Add to selection", "Replace selection"],
        index=0,
        key="chart_selection_mode",
        help="Add: each click appends more items. Replace: each click replaces current chart selection.",
    )
    if st.button("Clear Chart Selections", use_container_width=True):
        st.session_state["chart_filters"] = {}
        st.rerun()

charts_base_df = filtered_df.copy()
filtered_df = _apply_chart_filters(
    filtered_df,
    chart_filters,
    fuel_display_col=fuel_display_col,
    technology_display_col=technology_display_col,
    developer_col=developer_col,
    zone_col=zone_col,
    cod_col=cod_col,
)

if chart_filters:
    chart_filter_labels: list[str] = []
    if "fuel" in chart_filters:
        chart_filter_labels.append("Fuel (chart)")
    if "technology" in chart_filters:
        chart_filter_labels.append("Technology (chart)")
    if "developer" in chart_filters:
        chart_filter_labels.append("Developer (chart)")
    if "zone" in chart_filters:
        chart_filter_labels.append("Zone (chart)")
    if "cod_year" in chart_filters:
        chart_filter_labels.append("COD Year (chart)")
    active_filters.extend(chart_filter_labels)


st.subheader("Current Snapshot")
metric_cols = st.columns(4)
metric_cols[0].metric("Filtered Projects", f"{len(filtered_df):,}")
metric_cols[1].metric("Total Projects", f"{len(current_df):,}")

capacity_col = semantic.get("capacity_mw")
filtered_capacity_mw = None
if capacity_col and capacity_col in filtered_df.columns:
    filtered_capacity_mw = float(pd.to_numeric(filtered_df[capacity_col], errors="coerce").sum(skipna=True))
if official_under_study_mw is not None:
    metric_cols[2].metric("ERCOT Under Study (MW)", f"{official_under_study_mw:,.0f}")
elif filtered_capacity_mw is not None:
    metric_cols[2].metric("Filtered Capacity (MW)", f"{filtered_capacity_mw:,.0f}")
else:
    metric_cols[2].metric("Filtered Capacity (MW)", "n/a")

pulled_at_text = current_meta.get("pulled_at_utc", "Unknown") if current_meta else "Unknown"
metric_cols[3].metric("Last Pull (UTC)", _format_timestamp(pulled_at_text))

if active_filters:
    st.caption("Active filters: " + ", ".join(active_filters))
else:
    st.caption("Active filters: none")

if scope_notice:
    st.caption(scope_notice)

if official_under_study_mw is not None:
    if detail_project_capacity_mw is not None:
        difference_mw = official_under_study_mw - detail_project_capacity_mw
        st.caption(
            "ERCOT Summary total under study: "
            f"{official_under_study_mw:,.2f} MW. "
            "Project Details tabs total: "
            f"{detail_project_capacity_mw:,.2f} MW. "
            f"Difference: {difference_mw:,.2f} MW."
        )
        st.caption(
            "Note: Some under-study projects are counted in ERCOT summary totals but are not listed in the "
            "Project Details tabs yet (detail-level list excludes those undelivered/unlisted projects)."
        )
    else:
        st.caption(
            f"ERCOT Summary total under study: {official_under_study_mw:,.2f} MW "
            "(from the Summary tab)."
        )
    if selected_scope != "Combined (Large + Small)":
        st.caption("Scope filters adjust project-level views, while ERCOT Under Study remains the full system total.")

if current_meta:
    st.caption(
        "Source: "
        f"{current_meta.get('source', 'unknown')} | "
        f"Report Type ID: {current_meta.get('report_type_id', 'n/a')} | "
        f"Tabs Processed: {current_meta.get('tab_count', 'n/a')}"
    )
    with st.expander("Source Details", expanded=False):
        st.write(f"Data Product URL: {current_meta.get('data_product_url', 'n/a')}")
        st.write(f"Latest GIS URL: {current_meta.get('source_url', 'n/a')}")
        tabs_processed = current_meta.get("tabs_processed")
        if isinstance(tabs_processed, list) and tabs_processed:
            st.write("Tabs Processed: " + ", ".join(tabs_processed))

st.subheader("Fuel and Technology Mix")
st.caption("Chart clicks can select multiple items when `Chart Selection Mode` is set to `Add to selection`.")
chart_col_1, chart_col_2 = st.columns(2)
fuel_chart_df = _apply_chart_filters(
    charts_base_df,
    chart_filters,
    fuel_display_col=fuel_display_col,
    technology_display_col=technology_display_col,
    developer_col=developer_col,
    zone_col=zone_col,
    cod_col=cod_col,
    exclude_key="fuel",
)
technology_chart_df = _apply_chart_filters(
    charts_base_df,
    chart_filters,
    fuel_display_col=fuel_display_col,
    technology_display_col=technology_display_col,
    developer_col=developer_col,
    zone_col=zone_col,
    cod_col=cod_col,
    exclude_key="technology",
)

if fuel_display_col and fuel_display_col in fuel_chart_df.columns:
    fuel_plot = (
        fuel_chart_df.groupby(fuel_display_col, dropna=False)
        .size()
        .reset_index(name="projects")
        .sort_values("projects", ascending=False)
        .head(20)
    )
    fuel_plot, fuel_color_col = _apply_selection_highlight(
        fuel_plot,
        fuel_display_col,
        chart_filters.get("fuel", []),
    )
    fuel_chart_kwargs: dict[str, Any] = {}
    if fuel_color_col:
        fuel_chart_kwargs = {
            "color": fuel_color_col,
            "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
            "category_orders": {fuel_color_col: ["Selected", "Other"]},
        }
    fuel_fig = px.bar(
        fuel_plot,
        x=fuel_display_col,
        y="projects",
        title="Projects by Fuel",
        **fuel_chart_kwargs,
    )
    fuel_fig.update_layout(clickmode="event+select")
    _style_chart(fuel_fig)
    fuel_event = chart_col_1.plotly_chart(
        fuel_fig,
        use_container_width=True,
        key="fuel_chart",
        on_select="rerun",
    )
    fuel_selected = _selected_values_from_event(fuel_event, "x")
    if _update_chart_filter("fuel", fuel_selected):
        st.rerun()
else:
    chart_col_1.info("No fuel column detected for fuel chart.")

if technology_display_col and technology_display_col in technology_chart_df.columns:
    technology_count_plot = (
        technology_chart_df.groupby(technology_display_col, dropna=False)
        .size()
        .reset_index(name="projects")
        .sort_values("projects", ascending=False)
        .head(20)
    )
    technology_count_plot, technology_count_color_col = _apply_selection_highlight(
        technology_count_plot,
        technology_display_col,
        chart_filters.get("technology", []),
    )
    technology_chart_kwargs: dict[str, Any] = {}
    if technology_count_color_col:
        technology_chart_kwargs = {
            "color": technology_count_color_col,
            "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
            "category_orders": {technology_count_color_col: ["Selected", "Other"]},
        }
    technology_count_fig = px.bar(
        technology_count_plot,
        x=technology_display_col,
        y="projects",
        title="Projects by Technology (Count)",
        **technology_chart_kwargs,
    )
    technology_count_fig.update_layout(clickmode="event+select")
    _style_chart(technology_count_fig)
    technology_count_event = chart_col_2.plotly_chart(
        technology_count_fig,
        use_container_width=True,
        key="technology_count_chart",
        on_select="rerun",
    )
    technology_count_selected = _selected_values_from_event(technology_count_event, "x")
    if _update_chart_filter("technology", technology_count_selected):
        st.rerun()
else:
    chart_col_2.info("No technology column detected for technology chart.")

st.subheader("Technology Capacity (MW)")
if technology_display_col and technology_display_col in technology_chart_df.columns:
    if capacity_col and capacity_col in technology_chart_df.columns:
        technology_mw_df = technology_chart_df.copy()
        technology_mw_df[capacity_col] = pd.to_numeric(technology_mw_df[capacity_col], errors="coerce")
        technology_mw_plot = (
            technology_mw_df.groupby(technology_display_col, dropna=False)[capacity_col]
            .sum(min_count=1)
            .reset_index()
            .sort_values(capacity_col, ascending=False)
            .head(20)
        )
        technology_mw_plot, technology_mw_color_col = _apply_selection_highlight(
            technology_mw_plot,
            technology_display_col,
            chart_filters.get("technology", []),
        )
        technology_mw_chart_kwargs: dict[str, Any] = {}
        if technology_mw_color_col:
            technology_mw_chart_kwargs = {
                "color": technology_mw_color_col,
                "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
                "category_orders": {technology_mw_color_col: ["Selected", "Other"]},
            }
        technology_mw_fig = px.bar(
            technology_mw_plot,
            x=technology_display_col,
            y=capacity_col,
            title="Capacity by Technology (MW)",
            labels={capacity_col: "MW"},
            **technology_mw_chart_kwargs,
        )
        technology_mw_fig.update_layout(clickmode="event+select")
        _style_chart(technology_mw_fig)
        technology_mw_event = st.plotly_chart(
            technology_mw_fig,
            use_container_width=True,
            key="technology_mw_chart",
            on_select="rerun",
        )
        technology_mw_selected = _selected_values_from_event(technology_mw_event, "x")
        if _update_chart_filter("technology", technology_mw_selected):
            st.rerun()
    else:
        st.info("Capacity (MW) column missing for technology MW chart.")
else:
    st.info("No technology column detected for technology MW chart.")

cod_chart_df = _apply_chart_filters(
    charts_base_df,
    chart_filters,
    fuel_display_col=fuel_display_col,
    technology_display_col=technology_display_col,
    developer_col=developer_col,
    zone_col=zone_col,
    cod_col=cod_col,
    exclude_key="cod_year",
)
cod_col = semantic.get("cod_date")
if cod_col and cod_col in cod_chart_df.columns:
    cod_data = cod_chart_df.copy()
    cod_data[cod_col] = pd.to_datetime(cod_data[cod_col], errors="coerce")
    cod_data = cod_data.dropna(subset=[cod_col])

    if not cod_data.empty:
        cod_data["cod_year"] = cod_data[cod_col].dt.year.astype("int64")
        cod_totals = (
            cod_data.groupby("cod_year", dropna=False)
            .size()
            .reset_index(name="projects")
            .sort_values("cod_year")
        )

        st.subheader("Totals by Expected COD")
        if capacity_col and capacity_col in cod_data.columns:
            cod_data[capacity_col] = pd.to_numeric(cod_data[capacity_col], errors="coerce")
            cod_capacity = (
                cod_data.groupby("cod_year", dropna=False)[capacity_col]
                .sum(min_count=1)
                .reset_index()
            )
            cod_totals = cod_totals.merge(cod_capacity, on="cod_year", how="left")
            cod_totals, cod_mw_color_col = _apply_selection_highlight(
                cod_totals,
                "cod_year",
                chart_filters.get("cod_year", []),
            )
            cod_mw_chart_kwargs: dict[str, Any] = {}
            if cod_mw_color_col:
                cod_mw_chart_kwargs = {
                    "color": cod_mw_color_col,
                    "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
                    "category_orders": {cod_mw_color_col: ["Selected", "Other"]},
                }

            cod_col_1, cod_col_2 = st.columns(2)
            cod_mw_fig = px.bar(
                cod_totals,
                x="cod_year",
                y=capacity_col,
                title="Total Capacity by Expected COD Year (MW)",
                labels={capacity_col: "MW", "cod_year": "Expected COD Year"},
                hover_data={"projects": True},
                **cod_mw_chart_kwargs,
            )
            cod_mw_fig.update_layout(clickmode="event+select")
            _style_chart(cod_mw_fig, x_tick_angle=0)
            cod_mw_event = cod_col_1.plotly_chart(
                cod_mw_fig,
                use_container_width=True,
                key="expected_cod_year_mw_chart",
                on_select="rerun",
            )
            cod_mw_selected = _selected_values_from_event(cod_mw_event, "x")
            if _update_chart_filter("cod_year", cod_mw_selected):
                st.rerun()

            cod_count_fig = px.bar(
                cod_totals,
                x="cod_year",
                y="projects",
                title="Total Projects by Expected COD Year",
                labels={"projects": "Projects", "cod_year": "Expected COD Year"},
                hover_data={capacity_col: ":,.2f"},
                **(
                    {
                        "color": cod_mw_color_col,
                        "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
                        "category_orders": {cod_mw_color_col: ["Selected", "Other"]},
                    }
                    if cod_mw_color_col
                    else {}
                ),
            )
            cod_count_fig.update_layout(clickmode="event+select")
            _style_chart(cod_count_fig, x_tick_angle=0)
            cod_count_event = cod_col_2.plotly_chart(
                cod_count_fig,
                use_container_width=True,
                key="expected_cod_year_projects_chart",
                on_select="rerun",
            )
            cod_count_selected = _selected_values_from_event(cod_count_event, "x")
            if _update_chart_filter("cod_year", cod_count_selected):
                st.rerun()

            cod_totals["cumulative_capacity_mw"] = cod_totals[capacity_col].fillna(0).cumsum()
            timeline_fig = px.line(
                cod_totals,
                x="cod_year",
                y="cumulative_capacity_mw",
                title="Cumulative Planned Capacity by Expected COD Year",
            )
        else:
            cod_totals, cod_count_color_col = _apply_selection_highlight(
                cod_totals,
                "cod_year",
                chart_filters.get("cod_year", []),
            )
            cod_count_chart_kwargs: dict[str, Any] = {}
            if cod_count_color_col:
                cod_count_chart_kwargs = {
                    "color": cod_count_color_col,
                    "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
                    "category_orders": {cod_count_color_col: ["Selected", "Other"]},
                }
            cod_count_fig = px.bar(
                cod_totals,
                x="cod_year",
                y="projects",
                title="Total Projects by Expected COD Year",
                labels={"projects": "Projects", "cod_year": "Expected COD Year"},
                **cod_count_chart_kwargs,
            )
            cod_count_fig.update_layout(clickmode="event+select")
            _style_chart(cod_count_fig, x_tick_angle=0)
            cod_count_event = st.plotly_chart(
                cod_count_fig,
                use_container_width=True,
                key="expected_cod_year_projects_chart",
                on_select="rerun",
            )
            cod_count_selected = _selected_values_from_event(cod_count_event, "x")
            if _update_chart_filter("cod_year", cod_count_selected):
                st.rerun()

            cod_totals["cumulative_projects"] = cod_totals["projects"].cumsum()
            timeline_fig = px.line(
                cod_totals,
                x="cod_year",
                y="cumulative_projects",
                title="Cumulative Projects by Expected COD Year",
            )

        timeline_fig.update_layout(clickmode="event+select")
        _style_chart(timeline_fig, x_tick_angle=0)
        timeline_event = st.plotly_chart(
            timeline_fig,
            use_container_width=True,
            key="timeline_chart",
            on_select="rerun",
        )
        timeline_selected = _selected_values_from_event(timeline_event, "x")
        if _update_chart_filter("cod_year", timeline_selected):
            st.rerun()


st.subheader("Developer Analysis (Top 15)")
dev_col = semantic.get("developer")
developer_chart_df = _apply_chart_filters(
    charts_base_df,
    chart_filters,
    fuel_display_col=fuel_display_col,
    technology_display_col=technology_display_col,
    developer_col=developer_col,
    zone_col=zone_col,
    cod_col=cod_col,
    exclude_key="developer",
)
if dev_col and dev_col in developer_chart_df.columns:
    dev_col_1, dev_col_2 = st.columns(2)

    # Top 15 by MW
    if capacity_col and capacity_col in developer_chart_df.columns:
        dev_mw = (
            developer_chart_df.groupby(dev_col)[capacity_col]
            .sum()
            .reset_index()
            .sort_values(capacity_col, ascending=False)
            .head(15)
        )
        dev_mw, dev_mw_color_col = _apply_selection_highlight(
            dev_mw,
            dev_col,
            chart_filters.get("developer", []),
        )
        dev_mw_fig = px.bar(
            dev_mw,
            x=capacity_col,
            y=dev_col,
            orientation="h",
            title="Top 15 Developers by Capacity (MW)",
            labels={capacity_col: "Total MW", dev_col: "Developer"},
            **(
                {
                    "color": dev_mw_color_col,
                    "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
                    "category_orders": {dev_mw_color_col: ["Selected", "Other"]},
                }
                if dev_mw_color_col
                else {}
            ),
        )
        dev_mw_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        dev_mw_fig.update_layout(clickmode="event+select")
        _style_chart(dev_mw_fig, x_tick_angle=0)
        dev_mw_event = dev_col_1.plotly_chart(
            dev_mw_fig,
            use_container_width=True,
            key="developer_mw_chart",
            on_select="rerun",
        )
        dev_mw_selected = _selected_values_from_event(dev_mw_event, "y")
        if _update_chart_filter("developer", dev_mw_selected):
            st.rerun()

    # Top 15 by Count
    dev_count = (
        developer_chart_df.groupby(dev_col)
        .size()
        .reset_index(name="project_count")
        .sort_values("project_count", ascending=False)
        .head(15)
    )
    dev_count, dev_count_color_col = _apply_selection_highlight(
        dev_count,
        dev_col,
        chart_filters.get("developer", []),
    )
    dev_count_fig = px.bar(
        dev_count,
        x="project_count",
        y=dev_col,
        orientation="h",
        title="Top 15 Developers by Project Count",
        labels={"project_count": "Number of Projects", dev_col: "Developer"},
        **(
            {
                "color": dev_count_color_col,
                "color_discrete_map": {"Selected": "#4C78A8", "Other": "#9AA4B2"},
                "category_orders": {dev_count_color_col: ["Selected", "Other"]},
            }
            if dev_count_color_col
            else {}
        ),
    )
    dev_count_fig.update_layout(yaxis={"categoryorder": "total ascending"})
    dev_count_fig.update_layout(clickmode="event+select")
    _style_chart(dev_count_fig, x_tick_angle=0)
    dev_count_event = dev_col_2.plotly_chart(
        dev_count_fig,
        use_container_width=True,
        key="developer_count_chart",
        on_select="rerun",
    )
    dev_count_selected = _selected_values_from_event(dev_count_event, "y")
    if _update_chart_filter("developer", dev_count_selected):
        st.rerun()
else:
    st.info("No developer column detected for developer analysis.")


st.subheader("Regional Analysis")
fuel_col_for_region = fuel_display_col
zone_chart_df = _apply_chart_filters(
    charts_base_df,
    chart_filters,
    fuel_display_col=fuel_display_col,
    technology_display_col=technology_display_col,
    developer_col=developer_col,
    zone_col=zone_col,
    cod_col=cod_col,
    exclude_key="zone",
)

if zone_col and zone_col in zone_chart_df.columns:
    if capacity_col and capacity_col in zone_chart_df.columns:
        # Prepare data for stacked bars
        zone_fuel_df = (
            zone_chart_df.groupby([zone_col, fuel_col_for_region or "Unknown"])[capacity_col]
            .sum()
            .reset_index()
            .sort_values(capacity_col, ascending=False)
        )

        zone_fig = px.bar(
            zone_fuel_df,
            x=zone_col,
            y=capacity_col,
            color=fuel_col_for_region or "Unknown",
            title="Capacity (MW) by Reporting Zone and Fuel",
            labels={capacity_col: "Total MW", zone_col: "Reporting Zone", fuel_col_for_region or "Unknown": "Fuel"},
            barmode="stack",
            custom_data=[zone_col, fuel_col_for_region or "Unknown"],
        )
        zone_fig.update_layout(clickmode="event+select")
        _style_chart(zone_fig)
        zone_event = st.plotly_chart(
            zone_fig,
            use_container_width=True,
            key="zone_chart",
            on_select="rerun",
        )
        zone_selected = _selected_customdata_values(zone_event, 0)
        if _update_chart_filter("zone", zone_selected):
            st.rerun()
        fuel_selected = _selected_customdata_values(zone_event, 1)
        if _update_chart_filter("fuel", fuel_selected):
            st.rerun()
    else:
        st.info("Capacity (MW) column missing for regional analysis.")
else:
    st.info("No reporting zone column detected for regional analysis.")


st.subheader("Filtered Queue Records")
queue_display_df = filtered_df
queue_start_column: str | None = None
for candidate in ("inr", "ginr", "gir"):
    if candidate in filtered_df.columns:
        queue_start_column = candidate
        break

if queue_start_column is None:
    queue_id_col = semantic.get("queue_id")
    if queue_id_col and queue_id_col in filtered_df.columns:
        queue_start_column = queue_id_col

if queue_start_column:
    start_idx = filtered_df.columns.get_loc(queue_start_column)
    if isinstance(start_idx, int):
        queue_display_df = filtered_df.iloc[:, start_idx:]

st.dataframe(queue_display_df, use_container_width=True, height=420)
st.download_button(
    label="Download Filtered Data (CSV)",
    data=queue_display_df.to_csv(index=False).encode("utf-8"),
    file_name="ercot_queue_filtered.csv",
    mime="text/csv",
)

st.subheader("What Changed on Last Refresh")
change_report = load_change_report(current_meta)
summary = current_meta.get("diff_summary", {}) if current_meta else {}
summary_cols = st.columns(4)
summary_cols[0].metric("Added", summary.get("added", 0))
summary_cols[1].metric("Removed", summary.get("removed", 0))
summary_cols[2].metric("Changed", summary.get("changed", 0))
summary_cols[3].metric("Unchanged", summary.get("unchanged", 0))

if change_report:
    added_df = pd.DataFrame(change_report.get("added_sample", []))
    removed_df = pd.DataFrame(change_report.get("removed_sample", []))
    changed_fields_df = pd.DataFrame(change_report.get("changed_field_details", []))

    with st.expander("Added Projects (sample)", expanded=False):
        if added_df.empty:
            st.write("No added rows in this refresh.")
        else:
            st.dataframe(added_df, use_container_width=True)

    with st.expander("Removed Projects (sample)", expanded=False):
        if removed_df.empty:
            st.write("No removed rows in this refresh.")
        else:
            st.dataframe(removed_df, use_container_width=True)

    with st.expander("Changed Fields (sample)", expanded=False):
        if changed_fields_df.empty:
            st.write("No updated rows in this refresh.")
        else:
            st.dataframe(changed_fields_df, use_container_width=True)


st.subheader("Snapshot History")
history = load_snapshot_history(limit=30)
if history:
    history_df = pd.DataFrame(history)
    show_cols = [
        col
        for col in [
            "snapshot_id",
            "pulled_at_utc",
            "row_count",
            "source",
            "source_url",
            "data_product_url",
            "report_type_id",
            "tab_count",
            "diff_summary",
        ]
        if col in history_df.columns
    ]
    st.dataframe(history_df[show_cols], use_container_width=True, height=280)
else:
    st.write("No snapshot history available yet.")
