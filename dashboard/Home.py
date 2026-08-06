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

from dashboard.utils.data import clear_data_cache, demand_source_path, load_engine, load_funnel
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


@st.cache_data(show_spinner=False)
def prepare_home_data(engine: pd.DataFrame, funnel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hotel_columns = ["ProductID", "Region", "HotelType Short"]
    hotel_master = engine[hotel_columns].drop_duplicates("ProductID")
    detail = funnel.merge(hotel_master, on="ProductID", how="left", validate="many_to_one")
    detail["checkin_date"] = pd.to_datetime(detail["checkin_date"], errors="coerce").dt.normalize()
    detail["search_volume"] = pd.to_numeric(detail["search_volume"], errors="coerce").fillna(0)
    detail["view_volume"] = pd.to_numeric(detail["view_volume"], errors="coerce").fillna(0)
    for column in ["Previous_Searches", "Search_Change"]:
        if column not in detail:
            detail[column] = np.nan
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    detail = detail.dropna(subset=["checkin_date"])

    daily = (
        detail.groupby("checkin_date", as_index=False)
        .agg(
            Searches=("search_volume", "sum"),
            Views=("view_volume", "sum"),
            Hotels=("ProductID", "nunique"),
        )
        .sort_values("checkin_date")
    )
    q40, q75, q90 = daily["Searches"].quantile([0.40, 0.75, 0.90])
    daily["Demand Level"] = np.select(
        [daily["Searches"].ge(q90), daily["Searches"].ge(q75), daily["Searches"].ge(q40)],
        ["Very High", "High", "Medium"],
        default="Low",
    )
    return detail, daily


def level_badge(level: str) -> str:
    class_name = str(level).lower().replace(" ", "-")
    return f'<span class="demand-badge {class_name}">{level}</span>'


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
                            st.success(
                                f"Added {int(merge_stats['new_events']):,} new demand events from "
                                f"{len(preview):,} uploaded rows. Complete history now contains "
                                f"{len(merged):,} rows."
                            )
                            st.rerun()


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

selected_daily = (
    trend_detail.groupby("checkin_date", as_index=False)
    .agg(Searches=("search_volume", "sum"), Views=("view_volume", "sum"), Hotels=("ProductID", "nunique"))
    .sort_values("checkin_date")
)
selected_row = selected_daily[selected_daily["checkin_date"].eq(selected_ts)]
selected_searches = float(selected_detail["search_volume"].sum())
previous_searches = (
    float(selected_detail["Previous_Searches"].sum(min_count=1))
    if selected_detail["Previous_Searches"].notna().any()
    else np.nan
)
search_change = (
    float(selected_detail["Search_Change"].sum(min_count=1))
    if selected_detail["Search_Change"].notna().any()
    else np.nan
)
change_pct = search_change / previous_searches if previous_searches and pd.notna(search_change) else np.nan

portfolio_level = daily.loc[daily["checkin_date"].eq(selected_ts), "Demand Level"]
demand_level = portfolio_level.iloc[0] if not portfolio_level.empty else "No data"
high_dates = int(daily["Demand Level"].isin(["Very High", "High"]).sum())
active_hotels = int(selected_detail["ProductID"].nunique())
rising_hotels = int(selected_detail["Search_Change"].gt(0).sum())
hotel_median = selected_detail["search_volume"].median() if not selected_detail.empty else 0
hotel_q75 = selected_detail["search_volume"].quantile(0.75) if not selected_detail.empty else 0
hotel_change_pct = np.where(
    selected_detail["Previous_Searches"].gt(0),
    selected_detail["Search_Change"] / selected_detail["Previous_Searches"],
    np.nan,
)
action_hotels = int(
    (
        (
            selected_detail["search_volume"].ge(hotel_q75)
            & pd.Series(hotel_change_pct, index=selected_detail.index).ge(0.25)
        )
        | (
            selected_detail["search_volume"].ge(hotel_median)
            & pd.Series(hotel_change_pct, index=selected_detail.index).ge(0.10)
        )
        | (
            selected_detail["Previous_Searches"].isna()
            & selected_detail["search_volume"].ge(hotel_q75)
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
k1.metric("Demand signal", date_signal, help="Change versus the previous upload")
k2.metric("Latest searches", f"{selected_searches:,.0f}")
k3.metric(
    "Change vs previous upload",
    "No baseline" if pd.isna(change_pct) else f"{change_pct:+.1%}",
    delta=None if pd.isna(search_change) else f"{search_change:+,.0f} searches",
)
k4.metric("Hotels rising", f"{rising_hotels:,} of {active_hotels:,}")
k5.metric("Hotels requiring action", f"{action_hotels:,}")


chart_col, dates_col = st.columns([2.15, 1])
with chart_col:
    st.subheader("Demand by check-in date")
    chart_data = selected_daily.melt(
        id_vars="checkin_date",
        value_vars=["Searches", "Views"],
        var_name="Metric",
        value_name="Volume",
    )
    base = (
        alt.Chart(chart_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("checkin_date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=-35)),
            y=alt.Y("Volume:Q", title="Volume", axis=alt.Axis(format="~s")),
            color=alt.Color(
                "Metric:N",
                title=None,
                scale=alt.Scale(domain=["Searches", "Views"], range=["#2563eb", "#16a34a"]),
            ),
            tooltip=[
                alt.Tooltip("checkin_date:T", title="Check-in", format="%d %b %Y"),
                "Metric:N",
                alt.Tooltip("Volume:Q", format=",.0f"),
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


st.subheader(f"Hotels to check — {selected_ts:%d %b %Y}")
st.caption("Prioritized by latest search demand for the selected check-in date. Inventory and parity are checked outside HCI.")

hotel_table = selected_detail.sort_values(
    ["search_volume", "view_volume", "ProductName"], ascending=[False, False, True]
).head(12).copy()
hotel_table["Priority"] = np.arange(1, len(hotel_table) + 1)
if not hotel_table.empty:
    hotel_q50 = hotel_table["search_volume"].quantile(0.50)
    hotel_q80 = hotel_table["search_volume"].quantile(0.80)
    hotel_table["Demand Level"] = np.select(
        [hotel_table["search_volume"].ge(hotel_q80), hotel_table["search_volume"].ge(hotel_q50)],
        ["Very High", "High"],
        default="Medium",
    )
    hotel_table["Why prioritized"] = np.select(
        [hotel_table["Demand Level"].eq("Very High"), hotel_table["Demand Level"].eq("High")],
        ["Highest search demand", "Strong search demand"],
        default="Demand to monitor",
    )
    hotel_table["Next step"] = "Check inventory → parity"
    hotel_table = hotel_table.rename(
        columns={"ProductName": "Hotel", "search_volume": "Searches", "view_volume": "Views"}
    )
    st.dataframe(
        style_table(hotel_table[
            ["Priority", "Hotel", "Destination", "Demand Level", "Searches", "Views", "Why prioritized", "Next step"]
        ]),
        hide_index=True,
        width="stretch",
        height=455,
        column_config={
            "Priority": st.column_config.NumberColumn("Priority", format="%d", width="small"),
            "Searches": st.column_config.NumberColumn("Searches", format="localized"),
            "Views": st.column_config.NumberColumn("Views", format="localized"),
            "Next step": st.column_config.TextColumn("Next step", width="medium"),
        },
    )
else:
    st.info("No hotel demand is available for this date and filter selection.")

st.markdown(
    '<div class="workflow-note"><b>MM workflow:</b> Select a check-in date → identify high-demand hotels → '
    'check inventory manually → use the existing rate-parity tool.</div>',
    unsafe_allow_html=True,
)
