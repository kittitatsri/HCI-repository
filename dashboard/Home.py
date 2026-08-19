from __future__ import annotations

from datetime import date
import html
import json
from pathlib import Path
import shutil
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data import (
    build_checkin_date_trend,
    build_full_history_funnel,
    build_weekly_comparison,
    clear_data_cache,
    demand_source_path,
    iso_week_options,
    load_demand,
    load_engine,
    load_funnel,
)
from dashboard.utils.checkin_week import build_checkin_week_comparison
from dashboard.utils.incremental import merge_incremental_demand
from dashboard.utils.ui import apply_theme, style_table
from scripts.pipeline import RAW_DIR, REQUIRED_DEMAND_COLUMNS, run_pipeline


DEMAND_FILENAMES = ("demand_latest.csv.gz", "demand_latest.csv")


st.set_page_config(
    page_title="HCI | Home",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

update_message = st.session_state.pop("demand_update_message", None)
if update_message:
    st.success(update_message["success"])
    if update_message["skipped"]:
        st.warning(
            f"Skipped {update_message['skipped']:,} rows because ProductID was missing. "
            "The remaining rows were processed normally."
        )


@st.cache_data(show_spinner=False)
def prepare_home_data(engine: pd.DataFrame, funnel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hotel_columns = ["ProductID", "Region", "HotelType Short"]
    hotel_master = engine[hotel_columns].drop_duplicates("ProductID")
    detail = funnel.merge(hotel_master, on="ProductID", how="left", validate="many_to_one")
    detail["checkin_date"] = pd.to_datetime(detail["checkin_date"], errors="coerce").dt.normalize()
    detail["view_volume"] = pd.to_numeric(detail["view_volume"], errors="coerce").fillna(0)
    for column in [
        "Destination_Searches",
        "Previous_Destination_Searches",
        "Destination_Search_Change",
        "Previous_Views",
        "View_Change",
    ]:
        if column not in detail:
            detail[column] = np.nan
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    detail = detail.dropna(subset=["checkin_date"])
    daily = build_daily_market(detail)
    q40, q75, q90 = daily["Searches"].quantile([0.40, 0.75, 0.90])
    daily["Demand Level"] = np.select(
        [daily["Searches"].ge(q90), daily["Searches"].ge(q75), daily["Searches"].ge(q40)],
        ["Very High", "High", "Medium"],
        default="Low",
    )
    return detail, daily


def build_daily_market(detail: pd.DataFrame) -> pd.DataFrame:
    destination = detail.drop_duplicates(["Destination", "checkin_date"])
    searches = destination.groupby("checkin_date", as_index=False).agg(
        Searches=("Destination_Searches", "sum")
    )
    hotels = detail.groupby("checkin_date", as_index=False).agg(
        Views=("view_volume", "sum"), Hotels=("ProductID", "nunique")
    )
    return searches.merge(hotels, on="checkin_date", how="outer").sort_values("checkin_date")


def level_badge(level: str) -> str:
    class_name = str(level).lower().replace(" ", "-")
    return f'<span class="demand-badge {class_name}">{level}</span>'


def render_thailand_map(province_map: dict) -> None:
    """Render a fixed, Thailand-only SVG choropleth with cursor tooltips."""
    width, height, padding = 560, 620, 12
    points: list[tuple[float, float]] = []

    def polygon_rings(geometry: dict) -> list[list[list[float]]]:
        if geometry["type"] == "Polygon":
            return geometry["coordinates"]
        return [ring for polygon in geometry["coordinates"] for ring in polygon]

    for feature in province_map["features"]:
        for ring in polygon_rings(feature["geometry"]):
            points.extend((float(point[0]), float(point[1])) for point in ring)
    min_lon = min(point[0] for point in points)
    max_lon = max(point[0] for point in points)
    min_lat = min(point[1] for point in points)
    max_lat = max(point[1] for point in points)
    lon_scale = np.cos(np.deg2rad((min_lat + max_lat) / 2))
    scale = min(
        (width - 2 * padding) / ((max_lon - min_lon) * lon_scale),
        (height - 2 * padding) / (max_lat - min_lat),
    )
    map_width = (max_lon - min_lon) * lon_scale * scale
    map_height = (max_lat - min_lat) * scale
    offset_x = (width - map_width) / 2
    offset_y = (height - map_height) / 2

    def project(point: list[float]) -> tuple[float, float]:
        x = offset_x + (float(point[0]) - min_lon) * lon_scale * scale
        y = offset_y + (max_lat - float(point[1])) * scale
        return x, y

    paths = []
    for feature in province_map["features"]:
        properties = feature["properties"]
        commands = []
        for ring in polygon_rings(feature["geometry"]):
            projected = [project(point) for point in ring]
            if not projected:
                continue
            commands.append(
                "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in projected) + " Z"
            )
        color = properties["Color"]
        fill = f"rgb({color[0]},{color[1]},{color[2]})"
        destination = html.escape(str(properties["Destination"]), quote=True)
        demand_level = html.escape(str(properties["DemandLevel"]), quote=True)
        searches = int(properties["Searches"])
        views = int(properties["Views"])
        paths.append(
            f'<path d="{" ".join(commands)}" fill="{fill}" data-destination="{destination}" '
            f'data-demand="{demand_level}" data-searches="{searches:,}" data-views="{views:,}"/>'
        )

    map_html = f"""
    <div class="map-wrap">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Thailand demand map">
        {''.join(paths)}
      </svg>
      <div id="map-tooltip"></div>
    </div>
    <style>
      html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
      .map-wrap {{ position:relative; width:100%; height:450px; }}
      svg {{ width:100%; height:100%; display:block; }}
      path {{ stroke:#fff; stroke-width:1.15; vector-effect:non-scaling-stroke; cursor:default; transition:opacity .12s, stroke .12s; }}
      path:hover {{ opacity:.78; stroke:#0f172a; stroke-width:2.2; }}
      #map-tooltip {{ display:none; position:absolute; pointer-events:none; z-index:10; background:#0f172a; color:#fff; padding:9px 11px; border-radius:7px; font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; box-shadow:0 5px 18px rgba(15,23,42,.25); line-height:1.45; white-space:nowrap; }}
    </style>
    <script>
      const tooltip = document.getElementById('map-tooltip');
      document.querySelectorAll('path').forEach(path => {{
        path.addEventListener('mousemove', event => {{
          tooltip.innerHTML = `<b>${{path.dataset.destination}}</b><br>Demand: ${{path.dataset.demand}}<br>Searches: ${{path.dataset.searches}}<br>Hotel views: ${{path.dataset.views}}`;
          tooltip.style.display = 'block';
          const box = document.querySelector('.map-wrap').getBoundingClientRect();
          tooltip.style.left = Math.min(event.clientX - box.left + 14, box.width - tooltip.offsetWidth - 8) + 'px';
          tooltip.style.top = Math.max(event.clientY - box.top - 15, 8) + 'px';
        }});
        path.addEventListener('mouseleave', () => tooltip.style.display = 'none');
      }});
    </script>
    """
    components.html(map_html, height=455, scrolling=False)


def render_checkin_week(detail: pd.DataFrame, week_start: pd.Timestamp) -> None:
    week_end = week_start + pd.Timedelta(days=6)
    previous_start = week_start - pd.Timedelta(days=7)
    destinations = st.multiselect(
        "Destination",
        sorted(detail["Destination"].dropna().astype(str).unique()),
        placeholder="All destinations",
        key="checkin_week_destinations",
    )
    filtered = detail if not destinations else detail[detail["Destination"].isin(destinations)]
    hotels, destination_week, weekday = build_checkin_week_comparison(filtered, week_start)
    region_by_product = filtered.drop_duplicates("ProductID").set_index("ProductID")["Region"]
    hotels["Region"] = hotels["ProductID"].map(region_by_product)

    current_total = float(destination_week["Current_Searches"].sum())
    previous_total = float(destination_week["Previous_Searches"].sum())
    current_days = int(weekday.loc[weekday["Period"].eq("Selected week"), "checkin_date"].nunique())
    previous_days = int(weekday.loc[weekday["Period"].eq("Previous week"), "checkin_date"].nunique())
    complete = current_days == 7 and previous_days == 7
    current_basis = current_total if complete else current_total / current_days if current_days else np.nan
    previous_basis = previous_total if complete else previous_total / previous_days if previous_days else np.nan
    change = current_basis - previous_basis if pd.notna(current_basis) and pd.notna(previous_basis) else np.nan
    change_pct = change / previous_basis if previous_basis and pd.notna(change) else np.nan
    signal = (
        "No baseline" if pd.isna(change_pct) else
        "Critical surge" if change_pct >= 0.25 else
        "High increase" if change_pct >= 0.10 else
        "Declining" if change_pct <= -0.10 else "Stable"
    )

    hotels["Destination Signal"] = np.select(
        [
            hotels["Previous_Searches"].eq(0) & hotels["Current_Searches"].gt(0),
            hotels["Change_Pct"].ge(0.25),
            hotels["Change_Pct"].ge(0.10),
            hotels["Change_Pct"].le(-0.10),
        ],
        ["New demand", "Critical surge", "High increase", "Declining"],
        default="Stable",
    )
    rising_hotels = int(hotels["View_Change"].gt(0).sum())
    hotel_q75 = hotels["Current_Views"].quantile(0.75) if not hotels.empty else 0
    hotels["Action"] = (
        hotels["Destination Signal"].isin(["Critical surge", "High increase", "New demand"])
        & (hotels["View_Change"].gt(0) | hotels["Current_Views"].ge(hotel_q75))
    )

    st.caption(
        f"Check-in week: {week_start:%d %b}–{week_end:%d %b %Y} · "
        f"Previous: {previous_start:%d %b}–{week_start - pd.Timedelta(days=1):%d %b %Y}"
    )
    if not complete:
        st.warning(
            f"Incomplete comparison: selected week has {current_days}/7 dates and previous week has "
            f"{previous_days}/7. Change uses average searches per available day."
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    definition = (
        "Is demand for this check-in week higher or lower than demand for last check-in week? "
        "Compares check-in dates in the selected ISO week with dates in the previous ISO week."
    )
    k1.metric("Check-in week demand signal", signal, help=definition)
    k2.metric("Selected week searches", f"{current_total:,.0f}", help=definition)
    k3.metric(
        "Change vs previous check-in week",
        "No baseline" if pd.isna(change_pct) else f"{change_pct:+.1%}",
        delta=None if pd.isna(change) else f"{change:+,.0f} {'searches/day' if not complete else 'searches'}",
        help=definition,
    )
    k4.metric("Hotels with higher weekly views", f"{rising_hotels:,} of {len(hotels):,}")
    k5.metric("Hotels requiring action", f"{int(hotels['Action'].sum()):,}")

    chart_col, date_col = st.columns([2.15, 1])
    with chart_col:
        st.subheader("Searches and views: this week vs previous week", help=definition)
        weekly_chart = weekday.melt(
            id_vars=["checkin_date", "Matched_Date", "Period", "Weekday"],
            value_vars=["Searches", "Views"], var_name="Metric", value_name="Volume",
        )
        weekly_chart["Date Label"] = (
            weekly_chart["Matched_Date"].dt.day.astype(str)
            + " "
            + weekly_chart["Matched_Date"].dt.strftime("%b")
        )
        date_order = [
            f"{day.day} {day:%b}"
            for day in pd.date_range(week_start, week_end, freq="D")
        ]
        st.markdown(
            '<div style="display:flex;gap:22px;align-items:center;margin:0 0 4px 4px;'
            'font-size:0.86rem;color:#475569"><span><i style="display:inline-block;width:30px;'
            'border-top:3px solid #2563eb;margin-right:7px;vertical-align:middle"></i>'
            '<b>Selected week</b></span><span><i style="display:inline-block;width:30px;'
            'border-top:3px dashed #93c5fd;margin-right:7px;vertical-align:middle"></i>'
            '<b>Previous week</b></span></div>',
            unsafe_allow_html=True,
        )
        chart = (
            alt.Chart(weekly_chart)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("Date Label:N", sort=date_order, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Volume:Q", title="Observed value", axis=alt.Axis(format="~s")),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(
                        domain=["Searches", "Views"], range=["#2563eb", "#16a34a"]
                    ),
                    title=None,
                ),
                strokeDash=alt.StrokeDash(
                    "Period:N",
                    scale=alt.Scale(
                        domain=["Selected week", "Previous week"],
                        range=[[1, 0], [6, 4]],
                    ),
                    legend=None,
                ),
                tooltip=[
                    "Period:N", "Metric:N",
                    alt.Tooltip("Matched_Date:T", title="Date", format="%d %b %Y"),
                    alt.Tooltip("Volume:Q", format=",.0f"),
                ],
            )
            .properties(height=310)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Blue = searches · Green = views · Solid = selected week · Dotted = previous week.")
    with date_col:
        st.subheader("Strongest dates")
        strongest = (
            weekday[weekday["Period"].eq("Selected week")]
            .sort_values("Searches", ascending=False)
            .head(5)
        )
        for row in strongest.itertuples(index=False):
            st.markdown(
                f'<div class="date-row"><div><b>{row.checkin_date:%d %b %Y}</b><br>'
                f'<span>{row.Searches:,.0f} destination searches</span></div></div>',
                unsafe_allow_html=True,
            )

    st.subheader(f"Hotels to check — {week_start:%d %b}–{week_end:%d %b %Y}")
    hotel_order = {"Critical surge": 5, "High increase": 4, "New interest": 3, "Stable": 2, "Declining": 1}
    destination_order = {"Critical surge": 5, "High increase": 4, "New demand": 3, "Stable": 2, "Declining": 1}
    hotels["Hotel Signal"] = np.select(
        [
            hotels["Previous_Views"].isna(),
            hotels["View_Change_Pct"].ge(0.25),
            hotels["View_Change_Pct"].ge(0.10),
            hotels["View_Change_Pct"].le(-0.10),
        ],
        ["New interest", "Critical surge", "High increase", "Declining"],
        default="Stable",
    )
    hotels["Hotel Signal Order"] = hotels["Hotel Signal"].map(hotel_order).fillna(0)
    hotels["Destination Signal Order"] = hotels["Destination Signal"].map(destination_order).fillna(0)
    hotels["Has Views"] = hotels["Current_Views"].gt(0)
    hotels = hotels.sort_values(
        ["Has Views", "Current_Views", "Hotel Signal Order", "Destination Signal Order", "View_Change"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    hotels["Priority"] = np.arange(1, len(hotels) + 1)
    high_views = hotels["Current_Views"].gt(0) & hotels["Current_Views"].ge(
        hotels["Current_Views"].quantile(0.80)
    )
    rising_views = hotels["View_Change"].gt(0)
    hotels["Work Priority"] = np.select(
        [high_views & rising_views, high_views, rising_views],
        [
            "Urgent — high views and rising",
            "High — high views, even if stable",
            "Medium — lower views but rising",
        ],
        default="Routine — low or zero views without growth",
    )
    display = hotels.rename(
        columns={
            "ProductName": "Hotel",
            "Current_Views": "This Week Views",
            "View_Change_Pct": "View Change %",
        }
    )
    display["Product ID"] = display["ProductID"]
    display["View Change %"] = display["View Change %"] * 100
    st.dataframe(
        style_table(display[[
            "Priority", "Product ID", "Hotel", "Region", "Destination", "This Week Views",
            "View Change %", "Work Priority",
        ]].head(200)),
        hide_index=True,
        width="stretch",
        height=455,
        column_config={
            "Priority": st.column_config.NumberColumn(format="%d", width="small"),
            "Product ID": st.column_config.NumberColumn(format="%d", width="small"),
            "This Week Views": st.column_config.NumberColumn(format="localized"),
            "View Change %": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )


engine = load_engine()
funnel = load_funnel()
detail, daily = prepare_home_data(engine, funnel)


with st.sidebar:
    st.markdown('<div class="hci-brand"><b>HCI</b><span>HOTEL COMMERCIAL<br>INTELLIGENCE</span></div>', unsafe_allow_html=True)
    st.caption("Demand-led hotel prioritization")


header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("Home Dashboard")
    st.markdown("### Good morning 👋")
    st.caption("See where demand is strongest and which hotels to check first.")
with header_right:
    st.markdown('<div class="profile-card"><b>Market Manager</b></div>', unsafe_allow_html=True)


with st.expander("Update demand data", expanded=False):
    st.caption(
        "Upload only the newest daily export. HCI keeps the existing history, appends new events, "
        "and compares the updated state with the state before this upload."
    )
    upload = st.file_uploader("Upload the newest daily demand CSV or CSV.GZ", type=["csv", "gz"])
    if upload is not None:
        compressed = upload.name.lower().endswith(".csv.gz")
        try:
            preview = pd.read_csv(upload, compression="gzip" if compressed else None)
        except (UnicodeDecodeError, OSError) as exc:
            st.error("This file could not be read. Upload a valid CSV or GZIP-compressed CSV (.csv.gz).")
            st.caption(str(exc))
        else:
            upload.seek(0)
            missing = REQUIRED_DEMAND_COLUMNS.difference(preview.columns)
            if missing:
                st.error("Missing columns: " + ", ".join(sorted(missing)))
            else:
                preview_timestamp = pd.to_datetime(
                    preview["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False),
                    format="mixed",
                    errors="coerce",
                )
                if preview_timestamp.notna().any():
                    st.caption(
                        f"Uploaded period: {preview_timestamp.min():%d %b %Y, %H:%M} → "
                        f"{preview_timestamp.max():%d %b %Y, %H:%M} · {len(preview):,} rows"
                    )
            if not missing and st.button("Merge daily file and refresh", type="primary"):
                current_source = demand_source_path()
                try:
                    current_history = pd.read_csv(current_source)
                    merged, merge_stats = merge_incremental_demand(current_history, preview)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if not merge_stats["data_changed"]:
                        st.info("This daily file is already included. No data was changed.")
                    else:
                        destination = RAW_DIR / DEMAND_FILENAMES[0]
                        previous_destination = RAW_DIR / "demand_previous.csv.gz"
                        merged_temp = RAW_DIR / ".demand_latest.merge.csv.gz"
                        previous_temp = RAW_DIR / ".demand_previous.merge.csv.gz"
                        update_succeeded = False
                        try:
                            with st.spinner("Merging the daily snapshot and refreshing HCI…"):
                                current_history.to_csv(previous_temp, index=False, compression="gzip")
                                merged.to_csv(merged_temp, index=False, compression="gzip")
                                previous_temp.replace(previous_destination)
                                merged_temp.replace(destination)
                                (RAW_DIR / "demand_latest.csv").unlink(missing_ok=True)
                                (RAW_DIR / "demand_previous.csv").unlink(missing_ok=True)
                                try:
                                    run_pipeline(destination)
                                except Exception:
                                    shutil.copy2(previous_destination, destination)
                                    run_pipeline(destination)
                                    raise
                                update_succeeded = True
                        except Exception as exc:
                            st.error(f"The update failed and the previous demand state was restored: {exc}")
                        finally:
                            merged_temp.unlink(missing_ok=True)
                            previous_temp.unlink(missing_ok=True)
                        if update_succeeded:
                            clear_data_cache()
                            prepare_home_data.clear()
                            st.session_state["demand_update_message"] = {
                                "success": (
                                    f"Added {int(merge_stats['new_events']):,} new demand events from "
                                    f"{len(preview):,} uploaded rows. Complete history now contains "
                                    f"{len(merged):,} rows."
                                ),
                                "skipped": int(merge_stats["skipped_missing_product"]),
                            }
                            st.rerun()


mode_col, week_col = st.columns([1.2, 2.8])
view_mode = mode_col.radio(
    "Comparison view", ["Daily", "Snapshot Week", "Check-in Week"], horizontal=True
)
comparison_label = "previous upload"
if view_mode == "Snapshot Week":
    demand_history = load_demand()
    week_options = iso_week_options(demand_history)
    week_labels = [label for label, _ in week_options]
    selected_week_label = week_col.selectbox("ISO week", week_labels)
    selected_week_start = dict(week_options)[selected_week_label]
    weekly_funnel = build_weekly_comparison(demand_history, selected_week_start)
    selected_week_end = selected_week_start + pd.Timedelta(days=7)
    week_latest_snapshot = demand_history.loc[
        demand_history["snapshot_at"].ge(selected_week_start)
        & demand_history["snapshot_at"].lt(selected_week_end),
        "snapshot_at",
    ].max()
    week_col.caption(f"Latest snapshot available: {week_latest_snapshot:%d %b %Y, %H:%M}")
    detail, daily = prepare_home_data(engine, weekly_funnel)
    comparison_label = "previous ISO week"
elif view_mode == "Check-in Week":
    full_history_detail = build_full_history_funnel(load_demand())
    full_history_detail = full_history_detail[
        full_history_detail["ProductID"].isin(engine["ProductID"])
    ]
    full_history_detail, full_history_daily = prepare_home_data(engine, full_history_detail)
    week_starts = (
        full_history_daily["checkin_date"]
        - pd.to_timedelta(full_history_daily["checkin_date"].dt.weekday, unit="D")
    ).drop_duplicates().sort_values(ascending=False)
    week_options = {
        f"{start.isocalendar().year}-W{start.isocalendar().week:02d} "
        f"({start:%d %b}–{start + pd.Timedelta(days=6):%d %b})": start
        for start in week_starts
    }
    current_week_start = pd.Timestamp(date.today()) - pd.Timedelta(days=date.today().weekday())
    default_index = list(week_options.values()).index(current_week_start) if current_week_start in week_options.values() else 0
    selected_label = week_col.selectbox("Check-in ISO week", list(week_options), index=default_index)
    week_col.caption(
        "Compares demand for this check-in week with demand for the previous check-in week."
    )
    render_checkin_week(full_history_detail, pd.Timestamp(week_options[selected_label]))
    st.stop()
else:
    week_col.caption(
        "Daily compares summed intervals from the newest upload with an equally sized preceding window."
    )

available_dates = daily["checkin_date"].dt.date.tolist()
today = date.today()
default_date = today if today in available_dates else available_dates[0]

filter_1, filter_2, filter_3 = st.columns([1.2, 1.5, 2.3])
selected_date = filter_1.date_input(
    "Check-in date",
    value=default_date,
    min_value=min(available_dates),
    max_value=max(available_dates),
)
destinations = filter_2.multiselect(
    "Destination",
    sorted(detail["Destination"].dropna().astype(str).unique()),
    placeholder="All destinations",
)
filter_3.markdown(
    f'<div class="freshness">Data updated <b>{detail["snapshot_at"].max():%d %b %Y, %H:%M}</b></div>',
    unsafe_allow_html=True,
)

selected_ts = pd.Timestamp(selected_date)
selected_detail = detail[detail["checkin_date"].eq(selected_ts)].copy()
trend_detail = detail.copy()
if destinations:
    selected_detail = selected_detail[selected_detail["Destination"].isin(destinations)]
    trend_detail = trend_detail[trend_detail["Destination"].isin(destinations)]

selected_daily = build_daily_market(trend_detail)
if view_mode == "Daily":
    history_trend = build_checkin_date_trend(load_demand())
    if destinations:
        history_trend = history_trend[history_trend["Destination"].isin(destinations)]
    selected_daily_trend = (
        history_trend.groupby("checkin_date", as_index=False)[["Searches", "Views"]].sum()
        .sort_values("checkin_date")
    )
else:
    selected_daily_trend = selected_daily
selected_row = selected_daily[selected_daily["checkin_date"].eq(selected_ts)]
selected_destinations = selected_detail.drop_duplicates(["Destination", "checkin_date"]).copy()
selected_searches = float(selected_destinations["Destination_Searches"].sum())
previous_searches = (
    float(selected_destinations["Previous_Destination_Searches"].sum(min_count=1))
    if selected_destinations["Previous_Destination_Searches"].notna().any()
    else np.nan
)
search_change = (
    float(selected_destinations["Destination_Search_Change"].sum(min_count=1))
    if selected_destinations["Destination_Search_Change"].notna().any()
    else np.nan
)
change_pct = search_change / previous_searches if previous_searches and pd.notna(search_change) else np.nan

portfolio_level = daily.loc[daily["checkin_date"].eq(selected_ts), "Demand Level"]
demand_level = portfolio_level.iloc[0] if not portfolio_level.empty else "No data"
high_dates = int(daily["Demand Level"].isin(["Very High", "High"]).sum())
active_hotels = int(selected_detail["ProductID"].nunique())
rising_hotels = int(selected_detail["View_Change"].gt(0).sum())
hotel_q75 = selected_detail["view_volume"].quantile(0.75) if not selected_detail.empty else 0
selected_detail["View Change %"] = np.where(
    selected_detail["Previous_Views"].gt(0),
    selected_detail["View_Change"] / selected_detail["Previous_Views"],
    np.nan,
)
destination_change_pct = np.where(
    selected_destinations["Previous_Destination_Searches"].gt(0),
    selected_destinations["Destination_Search_Change"]
    / selected_destinations["Previous_Destination_Searches"],
    np.nan,
)
selected_destinations["Destination Signal"] = np.select(
    [
        selected_destinations["Previous_Destination_Searches"].isna(),
        pd.Series(destination_change_pct, index=selected_destinations.index).ge(0.25),
        pd.Series(destination_change_pct, index=selected_destinations.index).ge(0.10),
        pd.Series(destination_change_pct, index=selected_destinations.index).le(-0.10),
    ],
    ["New demand", "Critical surge", "High increase", "Declining"],
    default="Stable",
)
selected_detail = selected_detail.merge(
    selected_destinations[["Destination", "Destination Signal"]],
    on="Destination",
    how="left",
    validate="many_to_one",
)
action_hotels = int(
    (
        selected_detail["Destination Signal"].isin(["Critical surge", "High increase", "New demand"])
        & (
            selected_detail["View_Change"].gt(0)
            | selected_detail["view_volume"].ge(hotel_q75)
        )
    ).sum()
)
date_signal = (
    "No baseline"
    if pd.isna(change_pct)
    else "Critical surge"
    if change_pct >= 0.25
    else "High increase"
    if change_pct >= 0.10
    else "Declining"
    if change_pct <= -0.10
    else "Stable"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Destination demand signal",
    date_signal,
    help=(
        "How did our demand snapshot change from one upload week to another? Compares demand "
        "observed in the selected upload week with the previous upload week for the same "
        "check-in date."
        if view_mode == "Snapshot Week"
        else "Compares the latest demand upload with the previous upload for the selected check-in date."
    ),
)
k2.metric("Destination searches", f"{selected_searches:,.0f}")
k3.metric(
    "Change vs previous snapshot week" if view_mode == "Snapshot Week" else "Change vs previous upload",
    "No baseline" if pd.isna(change_pct) else f"{change_pct:+.1%}",
    delta=None if pd.isna(search_change) else f"{search_change:+,.0f} searches",
    help=(
        "How did our demand snapshot change from one upload week to another? Compares the same "
        "check-in date between the selected and previous upload weeks."
        if view_mode == "Snapshot Week"
        else "Compares the selected check-in date between the latest and previous demand uploads."
    ),
)
k4.metric("Hotels with rising views", f"{rising_hotels:,} of {active_hotels:,}")
k5.metric("Hotels requiring action", f"{action_hotels:,}")


st.subheader("Searches and views by check-in date")
trend_long = selected_daily_trend.melt(
    id_vars="checkin_date", value_vars=["Searches", "Views"],
    var_name="Metric", value_name="Volume",
)
base = (
    alt.Chart(trend_long)
    .mark_line(point=True, strokeWidth=2.5)
    .encode(
        x=alt.X("checkin_date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=-35)),
        y=alt.Y("Volume:Q", title="Observed value", axis=alt.Axis(format="~s")),
        color=alt.Color("Metric:N", title=None, scale=alt.Scale(
            domain=["Searches", "Views"], range=["#2563eb", "#16a34a"]
        )),
        tooltip=[
            alt.Tooltip("checkin_date:T", title="Check-in", format="%d %b %Y"),
            "Metric:N",
            alt.Tooltip("Volume:Q", title="Observed volume", format=",.0f"),
        ],
    )
    .properties(height=310)
)
selected_rule = (
    alt.Chart(pd.DataFrame({"checkin_date": [selected_ts]}))
    .mark_rule(color="#f59e0b", strokeWidth=2, strokeDash=[5, 4])
    .encode(x="checkin_date:T")
)
st.altair_chart(base + selected_rule, width="stretch")
st.caption("Y-axis shows actual observed searches and views. Hover over a point for the exact value.")


map_searches = selected_destinations[["Destination", "Destination_Searches"]].rename(
    columns={"Destination_Searches": "Searches"}
)
map_views = selected_detail.groupby("Destination", as_index=False)["view_volume"].sum().rename(
    columns={"view_volume": "Views"}
)
coordinates = pd.read_csv(ROOT / "data" / "reference" / "thailand_destination_coordinates.csv")
map_data = map_searches.merge(map_views, on="Destination", how="outer").merge(
    coordinates, on="Destination", how="inner", validate="one_to_one"
)
map_data[["Searches", "Views"]] = map_data[["Searches", "Views"]].fillna(0)
dates_col, map_col = st.columns([1, 2.25])
with dates_col:
    st.subheader("Strong demand dates")
    strongest = daily.sort_values(["Searches", "checkin_date"], ascending=[False, True]).head(5)
    for row in strongest.itertuples(index=False):
        st.markdown(
            f'<div class="date-row"><div><b>{row.checkin_date:%d %b %Y}</b><br>'
            f'<span>{row.Searches:,.0f} searches · {row.Hotels:,} hotels</span></div>'
            f'{level_badge(getattr(row, "_4"))}</div>',
            unsafe_allow_html=True,
        )
    st.caption(f"{high_dates} check-in dates currently show high or very high portfolio demand.")

with map_col:
    st.subheader("Demand Heatmap (Thailand)")
    if not map_data.empty:
        positive = map_data.loc[map_data["Searches"].gt(0), "Searches"]
        if positive.empty:
            q20 = q50 = q75 = q90 = 0
        else:
            q20, q50, q75, q90 = positive.quantile([0.20, 0.50, 0.75, 0.90])
        map_data["Demand Level"] = np.select(
            [
                map_data["Searches"].ge(q90),
                map_data["Searches"].ge(q75),
                map_data["Searches"].ge(q50),
                map_data["Searches"].ge(q20),
            ],
            ["Very High", "High", "Medium", "Low"],
            default="Very Low",
        )
        colors = {
            "Very High": [220, 38, 38, 210],
            "High": [249, 115, 22, 200],
            "Medium": [250, 204, 21, 190],
            "Low": [34, 197, 94, 170],
            "Very Low": [134, 239, 172, 150],
        }
        map_data["Color"] = map_data["Demand Level"].map(colors)
        demand_lookup = map_data.set_index("Destination").to_dict("index")
        province_aliases = {
            "Amnat Charoen": "Amnart Charoen",
            "Bangkok Metropolis": "Bangkok",
            "Bueng Kan": "Bungkan",
            "Buri Ram": "Buriram",
            "Chai Nat": "Chainat",
            "Chon Buri": "Chonburi",
            "Nakhon Ratchasima": "Nakhonratchasima",
            "Nong Bua Lam Phu": "Nong Bua Lamphu",
            "Phangnga": "Phang Nga",
            "Phatthalung": "Phattalung",
            "Prachuap Khiri Khan": "Prachuapkhirikhan",
        }
        with open(ROOT / "data" / "reference" / "thailand_provinces.geojson") as source:
            province_map = json.load(source)
        for feature in province_map["features"]:
            map_name = feature["properties"]["name"]
            destination = province_aliases.get(map_name, map_name)
            values = demand_lookup.get(
                destination,
                {"Searches": 0, "Views": 0, "Demand Level": "Very Low", "Color": colors["Very Low"]},
            )
            feature["properties"].update(
                {
                    "Destination": destination,
                    "Searches": int(round(values["Searches"])),
                    "Views": int(round(values["Views"])),
                    "DemandLevel": values["Demand Level"],
                    "Color": values["Color"],
                }
            )
        st.markdown(
            '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px;font-size:0.82rem;'
            'font-weight:600;color:#475569"><span style="color:#dc2626">● Very High</span>'
            '<span style="color:#f97316">● High</span><span style="color:#eab308">● Medium</span>'
            '<span style="color:#22c55e">● Low</span><span style="color:#86efac">● Very Low</span></div>',
            unsafe_allow_html=True,
        )
        render_thailand_map(province_map)
        st.caption(
            "Thailand only · Colour = relative destination search demand · "
            "Move the cursor over a province for its destination, searches, and hotel views."
        )
    else:
        st.info("No mapped destination demand is available for this date and filter selection.")


st.subheader(f"Hotels to check — {selected_ts:%d %b %Y}")
st.caption(
    f"Ranked first by observed hotel views in the comparison period; signals use the {comparison_label}. "
    "Inventory and parity are checked outside HCI."
)

selected_detail["Hotel Signal"] = np.select(
    [
        selected_detail["Previous_Views"].isna() | selected_detail["check_status"].eq("new entry"),
        selected_detail["View Change %"].ge(0.25),
        selected_detail["View Change %"].ge(0.10),
        selected_detail["View Change %"].le(-0.10),
    ],
    ["New interest", "Critical surge", "High increase", "Declining"],
    default="Stable",
)
hotel_signal_order = {"Critical surge": 5, "High increase": 4, "New interest": 3, "Stable": 2, "Declining": 1}
destination_signal_order = {"Critical surge": 5, "High increase": 4, "New demand": 3, "Stable": 2, "Declining": 1}
selected_detail["Hotel Signal Order"] = selected_detail["Hotel Signal"].map(hotel_signal_order).fillna(0)
selected_detail["Destination Signal Order"] = selected_detail["Destination Signal"].map(destination_signal_order).fillna(0)
selected_detail["Has Views"] = selected_detail["view_volume"].gt(0)
hotel_table = selected_detail.sort_values(
    ["Has Views", "view_volume", "Hotel Signal Order", "Destination Signal Order", "View_Change", "ProductName"],
    ascending=[False, False, False, False, False, True],
).head(12).copy()
hotel_table["Priority"] = np.arange(1, len(hotel_table) + 1)
if not hotel_table.empty:
    hotel_q50 = selected_detail["view_volume"].quantile(0.50)
    hotel_q80 = selected_detail["view_volume"].quantile(0.80)
    hotel_table["Demand Level"] = np.select(
        [hotel_table["view_volume"].ge(hotel_q80), hotel_table["view_volume"].ge(hotel_q50)],
        ["Very High", "High"],
        default="Medium",
    )
    hotel_table["Why prioritized"] = np.select(
        [
            hotel_table["Destination Signal"].isin(["Critical surge", "High increase"])
            & hotel_table["View_Change"].gt(0),
            hotel_table["Demand Level"].eq("Very High"),
        ],
        ["Destination rising + hotel views rising", "High hotel views"],
        default="Hotel interest to monitor",
    )
    high_views = hotel_table["view_volume"].gt(0) & hotel_table["view_volume"].ge(hotel_q80)
    rising_views = hotel_table["View_Change"].gt(0)
    hotel_table["Work Priority"] = np.select(
        [high_views & rising_views, high_views, rising_views],
        [
            "Urgent — high views and rising",
            "High — high views, even if stable",
            "Medium — lower views but rising",
        ],
        default="Routine — low or zero views without growth",
    )
    hotel_table = hotel_table.rename(
        columns={"ProductName": "Hotel", "view_volume": "Observed Views"}
    )
    hotel_table["Product ID"] = hotel_table["ProductID"]
    hotel_table["View Change %"] = hotel_table["View Change %"] * 100
    st.dataframe(
        style_table(hotel_table[
            [
                "Priority",
                "Product ID",
                "Hotel",
                "Region",
                "Destination",
                "Observed Views",
                "View Change %",
                "Why prioritized",
                "Work Priority",
            ]
        ]),
        hide_index=True,
        width="stretch",
        height=455,
        column_config={
            "Priority": st.column_config.NumberColumn("Priority", format="%d", width="small"),
            "Product ID": st.column_config.NumberColumn("Product ID", format="%d", width="small"),
            "Observed Views": st.column_config.NumberColumn("Observed Views", format="localized"),
            "View Change %": st.column_config.NumberColumn("View Change %", format="%+.1f%%"),
            "Work Priority": st.column_config.TextColumn("Work Priority", width="large"),
        },
    )
else:
    st.info("No hotel demand is available for this date and filter selection.")

st.markdown(
    '<div class="workflow-note"><b>MM workflow:</b> Select a check-in date → identify high-demand hotels → '
    'check inventory manually → use the existing rate-parity tool.</div>',
    unsafe_allow_html=True,
)
