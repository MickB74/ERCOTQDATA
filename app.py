from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from ercot_queue.config import MAX_CHANGE_SAMPLE_ROWS
from ercot_queue.diffing import calculate_diff
from ercot_queue.fetcher import fetch_latest_ercot_queue
from ercot_queue.processing import infer_semantic_columns, prepare_queue_dataframe
from ercot_queue.store import (
    ensure_data_dirs,
    load_change_report,
    load_latest_snapshot,
    load_snapshot_history,
    save_snapshot,
)


def _format_timestamp(raw_ts: str) -> str:
    try:
        parsed = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return raw_ts


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
    custom_url = st.text_input(
        "Custom ERCOT file URL (optional)",
        help="Leave blank to auto-discover the latest ERCOT GIS/interconnection report.",
    ).strip()

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

semantic = infer_semantic_columns(current_df)
filtered_df = current_df.copy()

with st.sidebar:
    st.header("Filters")

    for label, semantic_key in [("Status", "status"), ("Fuel / Technology", "fuel"), ("County", "county")]:
        column = semantic.get(semantic_key)
        if not column or column not in filtered_df.columns:
            continue

        options = sorted(
            {
                str(value)
                for value in filtered_df[column].dropna().astype(str)
                if str(value).strip()
            }
        )
        if not options:
            continue

        selected = st.multiselect(label, options=options, default=options)
        if selected:
            filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected)]

    capacity_col = semantic.get("capacity_mw")
    if capacity_col and capacity_col in filtered_df.columns:
        numeric_capacity = pd.to_numeric(filtered_df[capacity_col], errors="coerce")
        if numeric_capacity.notna().any():
            minimum = float(numeric_capacity.min())
            maximum = float(numeric_capacity.max())
            low, high = st.slider(
                "Capacity (MW)",
                min_value=minimum,
                max_value=maximum,
                value=(minimum, maximum),
            )
            filtered_df = filtered_df[numeric_capacity.between(low, high, inclusive="both")]

    cod_col = semantic.get("cod_date")
    if cod_col and cod_col in filtered_df.columns:
        cod_series = pd.to_datetime(filtered_df[cod_col], errors="coerce")
        if cod_series.notna().any():
            min_date = cod_series.min().date()
            max_date = cod_series.max().date()
            selected_dates = st.date_input("COD / In-Service Date", (min_date, max_date))
            if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
                mask = cod_series.between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
                filtered_df = filtered_df[mask]


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

chart_col_1, chart_col_2 = st.columns(2)

status_col = semantic.get("status")
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

fuel_col = semantic.get("fuel")
if fuel_col and fuel_col in filtered_df.columns:
    fuel_plot = (
        filtered_df.groupby(fuel_col, dropna=False)
        .size()
        .reset_index(name="projects")
        .sort_values("projects", ascending=False)
        .head(20)
    )
    fuel_fig = px.bar(fuel_plot, x=fuel_col, y="projects", title="Projects by Fuel / Technology")
    chart_col_2.plotly_chart(fuel_fig, use_container_width=True)
else:
    chart_col_2.info("No fuel/technology column detected for fuel chart.")

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
zone_col = semantic.get("reporting_zone")
fuel_col = semantic.get("fuel")

if zone_col and zone_col in filtered_df.columns:
    if capacity_col and capacity_col in filtered_df.columns:
        # Prepare data for stacked bars
        zone_fuel_df = (
            filtered_df.groupby([zone_col, fuel_col or "Unknown"])[capacity_col]
            .sum()
            .reset_index()
            .sort_values(capacity_col, ascending=False)
        )

        zone_fig = px.bar(
            zone_fuel_df,
            x=zone_col,
            y=capacity_col,
            color=fuel_col or "Unknown",
            title="Capacity (MW) by Reporting Zone and Technology",
            labels={capacity_col: "Total MW", zone_col: "Reporting Zone", fuel_col or "Unknown": "Technology"},
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
            "diff_summary",
        ]
        if col in history_df.columns
    ]
    st.dataframe(history_df[show_cols], use_container_width=True, height=280)
else:
    st.write("No snapshot history available yet.")
