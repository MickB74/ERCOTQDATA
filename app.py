import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Standardize ercot_queue package access
try:
    from ercot_queue.config import DEFAULT_DATA_PRODUCT_URLS, MAX_CHANGE_SAMPLE_ROWS
    from ercot_queue.diffing import calculate_diff
    from ercot_queue.fetcher import fetch_latest_ercot_queue
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
from ercot_queue.validation import (
    INTERCONNECTION_FYI_ERCOT_URL,
    compare_local_to_external,
    fetch_interconnection_fyi_ercot,
)

FUEL_CODE_MAP = {
    "BIO": "Biomass",
    "COA": "Coal",
    "GAS": "Gas",
    "GEO": "Geothermal",
    "HYD": "Hydrogen",
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
def _cached_fetch_external_validation(url: str) -> tuple[pd.DataFrame, dict]:
    return fetch_interconnection_fyi_ercot(url)


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


st.set_page_config(
    page_title="ERCOT Interconnection Queue",
    layout="wide",
)

ensure_data_dirs()

st.title("ERCOT Interconnection Queue Explorer")
st.caption(
    "Pull the latest queue data, filter it, visualize it, and track exactly what changed on every refresh."
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
        with st.spinner("Fetching and diffing latest data..."):
            previous_df, _ = load_latest_snapshot()
            latest_raw_df, source_meta = fetch_latest_ercot_queue(custom_url or None)
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
        st.success(
            "Refresh complete. "
            f"Added {diff_report['summary']['added']}, "
            f"Removed {diff_report['summary']['removed']}, "
            f"Changed {diff_report['summary']['changed']}."
        )


current_df, current_meta = load_latest_snapshot()
if current_df is None or current_df.empty:
    st.info("No snapshot yet. Click `Refresh Data` in the sidebar to pull the first dataset.")
    st.stop()

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
fuel_display_col = "fuel_name" if "fuel_name" in current_df.columns else fuel_col
technology_display_col = "technology_name" if "technology_name" in current_df.columns else technology_col

with st.sidebar:
    st.header("Filters")

    filter_definitions = [
        ("Status", status_col, "status"),
        ("Fuel", fuel_display_col, "fuel"),
        ("Technology", technology_display_col, "technology"),
        ("Reporting Zone", zone_col, "reporting_zone"),
        ("Developer", developer_col, "developer"),
        ("County", county_col, "county"),
    ]

    for label, column, filter_key in filter_definitions:
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

        # Add "Select All" logic
        select_all = st.checkbox(f"All {label}", value=True, key=f"all_{filter_key}")
        if not select_all:
            selected = st.multiselect(f"Select {label}", options=options, default=options, key=f"select_{filter_key}")
            if selected and set(selected) != set(options):
                filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected)]
                active_filters.append(label)
            elif not selected:
                # If nothing selected and not "Select All", show nothing
                filtered_df = filtered_df.iloc[0:0]
                active_filters.append(label)
            else:
                # "All" effectively selected, so no filter is applied.
                pass

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

    cod_col = semantic.get("cod_date")
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


st.subheader("Current Snapshot")
metric_cols = st.columns(4)
metric_cols[0].metric("Filtered Projects", f"{len(filtered_df):,}")
metric_cols[1].metric("Total Projects", f"{len(current_df):,}")

capacity_col = semantic.get("capacity_mw")
if capacity_col and capacity_col in filtered_df.columns:
    total_capacity = pd.to_numeric(filtered_df[capacity_col], errors="coerce").sum(skipna=True)
    metric_cols[2].metric("Filtered Capacity (MW)", f"{total_capacity:,.0f}")
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

if current_meta:
    st.caption(
        "Source: "
        f"{current_meta.get('source', 'unknown')} | "
        f"Report Type ID: {current_meta.get('report_type_id', 'n/a')} | "
        f"Tabs Processed: {current_meta.get('tab_count', 'n/a')}"
    )
    st.caption(f"Data Product URL: {current_meta.get('data_product_url', 'n/a')}")
    st.caption(f"Latest GIS URL: {current_meta.get('source_url', 'n/a')}")
    tabs_processed = current_meta.get("tabs_processed")
    if isinstance(tabs_processed, list) and tabs_processed:
        with st.expander("Tabs Processed In Latest Pull", expanded=False):
            st.write(", ".join(tabs_processed))

chart_col_1, chart_col_2 = st.columns(2)

if status_col and status_col in filtered_df.columns:
    status_data = filtered_df.copy()
    if capacity_col and capacity_col in status_data.columns:
        status_data[capacity_col] = pd.to_numeric(status_data[capacity_col], errors="coerce")
        plot_df = (
            status_data.groupby(status_col, dropna=False)[capacity_col]
            .sum(min_count=1)
            .reset_index()
            .sort_values(capacity_col, ascending=False)
        )
        y_axis = capacity_col
        title = "Capacity by Status"
    else:
        plot_df = (
            status_data.groupby(status_col, dropna=False)
            .size()
            .reset_index(name="projects")
            .sort_values("projects", ascending=False)
        )
        y_axis = "projects"
        title = "Projects by Status"

    status_fig = px.bar(plot_df, x=status_col, y=y_axis, title=title)
    chart_col_1.plotly_chart(status_fig, use_container_width=True)
else:
    chart_col_1.info("No status column detected for status chart.")

if fuel_display_col and fuel_display_col in filtered_df.columns:
    fuel_plot = (
        filtered_df.groupby(fuel_display_col, dropna=False)
        .size()
        .reset_index(name="projects")
        .sort_values("projects", ascending=False)
        .head(20)
    )
    fuel_fig = px.bar(fuel_plot, x=fuel_display_col, y="projects", title="Projects by Fuel")
    chart_col_2.plotly_chart(fuel_fig, use_container_width=True)
else:
    chart_col_2.info("No fuel column detected for fuel chart.")

st.subheader("Projects by Technology")
if technology_display_col and technology_display_col in filtered_df.columns:
    technology_plot = (
        filtered_df.groupby(technology_display_col, dropna=False)
        .size()
        .reset_index(name="projects")
        .sort_values("projects", ascending=False)
        .head(20)
    )
    technology_fig = px.bar(
        technology_plot,
        x=technology_display_col,
        y="projects",
        title="Projects by Technology",
    )
    st.plotly_chart(technology_fig, use_container_width=True)
else:
    st.info("No technology column detected for technology chart.")

cod_col = semantic.get("cod_date")
if cod_col and cod_col in filtered_df.columns:
    cod_data = filtered_df.copy()
    cod_data[cod_col] = pd.to_datetime(cod_data[cod_col], errors="coerce")
    cod_data = cod_data.dropna(subset=[cod_col])

    if not cod_data.empty:
        cod_data["cod_month"] = cod_data[cod_col].dt.to_period("M").dt.to_timestamp()
        if capacity_col and capacity_col in cod_data.columns:
            cod_data[capacity_col] = pd.to_numeric(cod_data[capacity_col], errors="coerce")
            timeline_df = (
                cod_data.groupby("cod_month", dropna=False)[capacity_col]
                .sum(min_count=1)
                .reset_index()
                .sort_values("cod_month")
            )
            timeline_df["cumulative_capacity_mw"] = timeline_df[capacity_col].fillna(0).cumsum()
            timeline_fig = px.line(
                timeline_df,
                x="cod_month",
                y="cumulative_capacity_mw",
                title="Cumulative Planned Capacity by COD",
            )
        else:
            timeline_df = (
                cod_data.groupby("cod_month", dropna=False)
                .size()
                .reset_index(name="projects")
                .sort_values("cod_month")
            )
            timeline_df["cumulative_projects"] = timeline_df["projects"].cumsum()
            timeline_fig = px.line(
                timeline_df,
                x="cod_month",
                y="cumulative_projects",
                title="Cumulative Projects by COD",
            )
        st.plotly_chart(timeline_fig, use_container_width=True)


st.subheader("Developer Analysis (Top 15)")
dev_col = semantic.get("developer")
if dev_col and dev_col in filtered_df.columns:
    dev_col_1, dev_col_2 = st.columns(2)

    # Top 15 by MW
    if capacity_col and capacity_col in filtered_df.columns:
        dev_mw = (
            filtered_df.groupby(dev_col)[capacity_col]
            .sum()
            .reset_index()
            .sort_values(capacity_col, ascending=False)
            .head(15)
        )
        dev_mw_fig = px.bar(
            dev_mw,
            x=capacity_col,
            y=dev_col,
            orientation="h",
            title="Top 15 Developers by Capacity (MW)",
            labels={capacity_col: "Total MW", dev_col: "Developer"},
        )
        dev_mw_fig.update_layout(yaxis={"categoryorder": "total ascending"})
        dev_col_1.plotly_chart(dev_mw_fig, use_container_width=True)

    # Top 15 by Count
    dev_count = (
        filtered_df.groupby(dev_col)
        .size()
        .reset_index(name="project_count")
        .sort_values("project_count", ascending=False)
        .head(15)
    )
    dev_count_fig = px.bar(
        dev_count,
        x="project_count",
        y=dev_col,
        orientation="h",
        title="Top 15 Developers by Project Count",
        labels={"project_count": "Number of Projects", dev_col: "Developer"},
    )
    dev_count_fig.update_layout(yaxis={"categoryorder": "total ascending"})
    dev_col_2.plotly_chart(dev_count_fig, use_container_width=True)
else:
    st.info("No developer column detected for developer analysis.")


st.subheader("Regional Analysis")
fuel_col_for_region = fuel_display_col

if zone_col and zone_col in filtered_df.columns:
    if capacity_col and capacity_col in filtered_df.columns:
        # Prepare data for stacked bars
        zone_fuel_df = (
            filtered_df.groupby([zone_col, fuel_col_for_region or "Unknown"])[capacity_col]
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
        )
        st.plotly_chart(zone_fig, use_container_width=True)
    else:
        st.info("Capacity (MW) column missing for regional analysis.")
else:
    st.info("No reporting zone column detected for regional analysis.")


st.subheader("Filtered Queue Records")
st.dataframe(filtered_df, use_container_width=True, height=420)
st.download_button(
    label="Download Filtered Data (CSV)",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
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


st.subheader("External Validation (Interconnection.fyi)")
st.caption(
    "Cross-check this snapshot against an independent source and highlight queue ID, status, and capacity mismatches."
)

validation_col_1, validation_col_2 = st.columns([3, 1])
with validation_col_1:
    validation_url = st.text_input(
        "Validation Source URL",
        value=INTERCONNECTION_FYI_ERCOT_URL,
        help="Independent source used for queue cross-checking.",
    ).strip()
with validation_col_2:
    run_validation = st.button("Run External Validation", use_container_width=True)
    if st.button("Clear Validation Results", use_container_width=True):
        if "external_validation" in st.session_state:
            del st.session_state["external_validation"]
        st.rerun()

if run_validation:
    try:
        with st.spinner("Fetching external source and comparing queue IDs..."):
            external_df, external_meta = _cached_fetch_external_validation(validation_url)
            validation_result = compare_local_to_external(
                current_df,
                external_df,
                local_queue_col=semantic.get("queue_id"),
                local_status_col=semantic.get("status"),
                local_capacity_col=semantic.get("capacity_mw"),
            )
        st.session_state["external_validation"] = {
            "ran_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "source_meta": external_meta,
            "result": validation_result,
        }
        st.success(
            "Validation complete. "
            f"Matched {validation_result['summary']['matched_queue_ids']} queue IDs; "
            f"missing local: {validation_result['summary']['missing_in_local']}, "
            f"missing external: {validation_result['summary']['missing_in_external']}."
        )
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"External validation failed: {exc}")

validation_state = st.session_state.get("external_validation")
if isinstance(validation_state, dict):
    val_res = validation_state.get("result", {})
    summary = val_res.get("summary", {})
    source_meta = validation_state.get("source_meta", {})
    
    # Only show if we have a valid summary structure
    if isinstance(summary, dict) and summary:
        st.caption(
            f"Last validation run: {validation_state.get('ran_at_utc', 'unknown')} UTC | "
            f"Parser: {source_meta.get('parser', 'unknown')} | "
            f"Source: {source_meta.get('source_url', validation_url)}"
        )

    # Collect all unique statuses for filtering
    all_statuses = set()
    for df_key in ["missing_in_local", "missing_in_external", "status_mismatches", "capacity_mismatches"]:
        df = val_res.get(df_key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            for col in ["status", "local_status", "external_status"]:
                if col in df.columns:
                    all_statuses.update(df[col].dropna().unique())
    
    sorted_statuses = sorted(list(all_statuses))
    
    st.markdown("---")
    val_filter_col1, val_filter_col2 = st.columns([1, 2])
    with val_filter_col1:
        st.write("**Filter Results by Status**")
        select_all_val = st.checkbox("All Statuses", value=True, key="val_status_all")
    with val_filter_col2:
        if select_all_val:
            selected_val_statuses = sorted_statuses
            st.multiselect("Statuses", options=sorted_statuses, default=sorted_statuses, disabled=True, key="val_status_sel_all")
        else:
            selected_val_statuses = st.multiselect("Select Statuses", options=sorted_statuses, default=sorted_statuses, key="val_status_sel")

    # Filter DataFrames
    def filter_val_df(df, statuses):
        if df is None or df.empty:
            return df
        mask = pd.Series(False, index=df.index)
        for col in ["status", "local_status", "external_status"]:
            if col in df.columns:
                mask |= df[col].isin(statuses)
        return df[mask]

    missing_in_local_df = filter_val_df(val_res["missing_in_local"], selected_val_statuses)
    missing_in_external_df = filter_val_df(val_res["missing_in_external"], selected_val_statuses)
    status_mismatch_df = filter_val_df(val_res["status_mismatches"], selected_val_statuses)
    capacity_mismatch_df = filter_val_df(val_res["capacity_mismatches"], selected_val_statuses)

    # Recalculate Summary for metrics (based on filtered DataFrames)
    filtered_summary = {
        "local_queue_ids": len(missing_in_external_df) + (summary.get("matched_queue_ids", 0) if select_all_val else 0), # This is approximate if not all matched are shown
        "external_queue_ids": len(missing_in_local_df) + (summary.get("matched_queue_ids", 0) if select_all_val else 0),
        "matched_queue_ids": summary.get("matched_queue_ids", 0) if select_all_val else "n/a",
        "missing_in_local": len(missing_in_local_df),
        "missing_in_external": len(missing_in_external_df),
        "status_mismatches": len(status_mismatch_df),
        "capacity_mismatches": len(capacity_mismatch_df),
        "missing_in_local_mw": missing_in_local_df["capacity_mw"].sum() if "capacity_mw" in missing_in_local_df.columns else 0,
        "missing_in_external_mw": missing_in_external_df["capacity_mw"].sum() if "capacity_mw" in missing_in_external_df.columns else 0,
        "status_mismatch_mw": status_mismatch_df["local_capacity_mw"].sum() if "local_capacity_mw" in status_mismatch_df.columns else (status_mismatch_df["capacity_mw"].sum() if "capacity_mw" in status_mismatch_df.columns else 0),
        "capacity_mismatch_mw": capacity_mismatch_df["local_capacity_mw"].sum() if "local_capacity_mw" in capacity_mismatch_df.columns else 0,
    }
    # Note: matched_mw and overall totals are hard to filter perfectly without re-running the comparison or having more data.
    # We will focus on the mismatch metrics which are most relevant for filtering.

    st.markdown("##### Filtered Mismatch Metrics")
    val_metrics_ids = st.columns(4)
    val_metrics_ids[0].metric("Missing Local", f"{filtered_summary['missing_in_local']:,}")
    val_metrics_ids[1].metric("Missing External", f"{filtered_summary['missing_in_external']:,}")
    val_metrics_ids[2].metric("Status Mismatches", f"{filtered_summary['status_mismatches']:,}")
    val_metrics_ids[3].metric("Capacity Mismatches", f"{filtered_summary['capacity_mismatches']:,}")

    val_metrics_mw = st.columns(4)
    val_metrics_mw[0].metric("Missing Local MW", f"{filtered_summary['missing_in_local_mw']:,.0f}")
    val_metrics_mw[1].metric("Missing External MW", f"{filtered_summary['missing_in_external_mw']:,.0f}")
    val_metrics_mw[2].metric("Status Mismatch MW", f"{filtered_summary['status_mismatch_mw']:,.0f}")
    val_metrics_mw[3].metric("Capacity Mismatch MW", f"{filtered_summary['capacity_mismatch_mw']:,.0f}")

    with st.expander("Queue IDs Missing In Local Snapshot", expanded=False):
        if not missing_in_local_df.empty:
            st.dataframe(missing_in_local_df, use_container_width=True)
        else:
            st.write("No missing queue IDs in local snapshot (for selected statuses).")

    with st.expander("Queue IDs Missing In External Source", expanded=False):
        if not missing_in_external_df.empty:
            st.dataframe(missing_in_external_df, use_container_width=True)
        else:
            st.write("No missing queue IDs in external source (for selected statuses).")

    with st.expander("Status Mismatches", expanded=False):
        if not status_mismatch_df.empty:
            st.dataframe(status_mismatch_df, use_container_width=True)
        else:
            st.write("No status mismatches detected (for selected statuses).")

    with st.expander("Capacity Mismatches (MW)", expanded=False):
        if not capacity_mismatch_df.empty:
            st.dataframe(capacity_mismatch_df, use_container_width=True)
        else:
            st.write("No capacity mismatches above tolerance (for selected statuses).")
