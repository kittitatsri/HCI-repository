from __future__ import annotations

from datetime import date
from pathlib import Path
import shutil
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data import (
    build_checkin_date_trend,
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
    hotel_columns = [
        "ProductID", "Region", "HotelType Short", "Agoda Status", "Ctrip Status"
    ]
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
        st.subheader("This check-in week vs previous week", help=definition)
        chart = (
            alt.Chart(weekday)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=alt.X("Weekday:O", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], title=None),
                y=alt.Y("Searches:Q", title="Destination searches", axis=alt.Axis(format="~s")),
                color=alt.Color(
                    "Period:N",
                    scale=alt.Scale(
                        domain=["Selected week", "Previous week"], range=["#2563eb", "#94a3b8"]
                    ),
                    title=None,
                ),
                tooltip=["Period:N", "Weekday:N", alt.Tooltip("Searches:Q", format=",.0f")],
            )
            .properties(height=310)
        )
        st.altair_chart(chart, width="stretch")
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
    display["View Change %"] = display["View Change %"] * 100
    st.dataframe(
        style_table(display[[
            "Priority", "Hotel", "Destination", "This Week Views",
            "View Change %", "Work Priority",
        ]].head(200)),
        hide_index=True,
        width="stretch",
        height=455,
        column_config={
            "Priority": st.column_config.NumberColumn(format="%d", width="small"),
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


available_dates = daily["checkin_date"].dt.date.tolist()
today = date.today()
default_date = today if today in available_dates else available_dates[0]

filter_1, filter_2, filter_3 = st.columns([1.2, 1.6, 2.2])
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
if destinations:
    selected_detail = selected_detail[selected_detail["Destination"].isin(destinations)]
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

active_hotels = int(selected_detail["ProductID"].nunique())
selected_detail["View Change %"] = np.where(
    selected_detail["Previous_Views"].gt(0),
    selected_detail["View_Change"] / selected_detail["Previous_Views"],
    np.nan,
)
hotel_q80 = selected_detail["view_volume"].quantile(0.80) if not selected_detail.empty else 0
high_views = selected_detail["view_volume"].gt(0) & selected_detail["view_volume"].ge(hotel_q80)
rising_views = selected_detail["View_Change"].gt(0)
selected_detail["Work Priority"] = np.select(
    [high_views & rising_views, high_views, rising_views],
    [
        "Urgent — high views and rising",
        "High — high views, even if stable",
        "Medium — lower views but rising",
    ],
    default="Routine — low or zero views without growth",
)
priority_order = {
    "Urgent — high views and rising": 1,
    "High — high views, even if stable": 2,
    "Medium — lower views but rising": 3,
    "Routine — low or zero views without growth": 4,
}
selected_detail["Priority Order"] = selected_detail["Work Priority"].map(priority_order)

urgent = int(selected_detail["Work Priority"].str.startswith("Urgent").sum())
high = int(selected_detail["Work Priority"].str.startswith("High").sum())
medium = int(selected_detail["Work Priority"].str.startswith("Medium").sum())
mapping_issue = int(
    (~selected_detail["Agoda Status"].eq("Mapped") | ~selected_detail["Ctrip Status"].eq("Mapped")).sum()
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Urgent", f"{urgent:,}", help="High hotel views and rising versus the previous upload.")
k2.metric("High priority", f"{high:,}", help="High hotel views, even when stable.")
k3.metric("Medium priority", f"{medium:,}", help="Lower hotel views that are rising.")
k4.metric("Hotels with demand", f"{active_hotels:,}")
k5.metric("Mapping checks", f"{mapping_issue:,}", help="Hotels not mapped on Agoda or Ctrip.")

if pd.isna(change_pct):
    st.info("Demand comparison is unavailable for this date. Prioritize using observed hotel views.")
elif change_pct >= 0.10:
    st.warning(
        f"Destination searches are {change_pct:+.1%} versus the previous upload "
        f"({search_change:+,.0f}). Review Urgent hotels first."
    )

queue = selected_detail.sort_values(
    ["Priority Order", "view_volume", "View_Change", "ProductName"],
    ascending=[True, False, False, True],
).head(10).copy()
queue["Priority"] = np.arange(1, len(queue) + 1)
queue["Hotel"] = queue["ProductName"]
queue["Observed Views"] = queue["view_volume"]
queue["View Change %"] = queue["View Change %"] * 100
queue["Open Hotel"] = queue.apply(
    lambda row: (
        f"Hotel_Explorer?product_id={int(row['ProductID'])}"
        f"&checkin_date={selected_ts:%Y-%m-%d}"
    ),
    axis=1,
)

st.subheader(f"Today’s hotel work queue — check-in {selected_ts:%d %b %Y}")
st.caption(
    "Start at the top. Check inventory manually, then verify rate parity and mapping in Hotel Explorer."
)
if not queue.empty:
    st.dataframe(
        style_table(queue[[
            "Priority", "Work Priority", "Hotel", "Destination", "Observed Views",
            "View Change %", "Open Hotel",
        ]]),
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Priority": st.column_config.NumberColumn(format="%d", width="small"),
            "Work Priority": st.column_config.TextColumn(width="large"),
            "Observed Views": st.column_config.NumberColumn(format="localized"),
            "View Change %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Open Hotel": st.column_config.LinkColumn("Action", display_text="Open Hotel Explorer"),
        },
    )
else:
    st.info("No hotel demand is available for this date and filter selection.")

nav_left, nav_right = st.columns(2)
nav_left.page_link("pages/1_📅_Demand_by_Date.py", label="Open Demand by Date for deeper analysis", icon="📅")
nav_right.page_link("pages/3_✅_Data_Quality.py", label="Review Data Quality", icon="✅")
