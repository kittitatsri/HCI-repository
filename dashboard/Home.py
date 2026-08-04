from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data import clear_data_cache, load_engine, load_funnel
from dashboard.utils.ui import apply_theme
from scripts.pipeline import DEMAND_FILENAMES, RAW_DIR, REQUIRED_DEMAND_COLUMNS, run_pipeline


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
    st.markdown("### Good morning, Kittitat 👋")
    st.caption("See where demand is strongest and which hotels to check first.")
with header_right:
    st.markdown('<div class="profile-card"><b>Kittitat Sri</b><br><span>Market Manager</span></div>', unsafe_allow_html=True)


with st.expander("Update demand data", expanded=False):
    upload = st.file_uploader("Upload the latest demand CSV or CSV.GZ", type=["csv", "gz"])
    if upload is not None:
        preview = pd.read_csv(upload)
        upload.seek(0)
        missing = REQUIRED_DEMAND_COLUMNS.difference(preview.columns)
        if missing:
            st.error("Missing columns: " + ", ".join(sorted(missing)))
        elif st.button("Process and refresh", type="primary"):
            compressed = upload.name.lower().endswith(".csv.gz")
            destination = RAW_DIR / (DEMAND_FILENAMES[0] if compressed else DEMAND_FILENAMES[1])
            alternative = RAW_DIR / (DEMAND_FILENAMES[1] if compressed else DEMAND_FILENAMES[0])
            destination.write_bytes(upload.getvalue())
            alternative.unlink(missing_ok=True)
            with st.spinner("Refreshing daily hotel demand…"):
                run_pipeline(destination)
            clear_data_cache()
            prepare_home_data.clear()
            st.success(f"Updated from {len(preview):,} demand rows.")
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
selected_views = float(selected_detail["view_volume"].sum())

portfolio_level = daily.loc[daily["checkin_date"].eq(selected_ts), "Demand Level"]
demand_level = portfolio_level.iloc[0] if not portfolio_level.empty else "No data"
high_dates = int(daily["Demand Level"].isin(["Very High", "High"]).sum())
active_hotels = int(selected_detail["ProductID"].nunique())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Demand level", demand_level, help="Based on this date's searches compared with other check-in dates")
k2.metric("Searches", f"{selected_searches:,.0f}", help="Latest cumulative searches for the selected check-in date")
k3.metric("Views", f"{selected_views:,.0f}", help="Latest cumulative views for the selected check-in date")
k4.metric("Hotels with demand", f"{active_hotels:,}", help="Hotels represented on the selected check-in date")


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
        hotel_table[
            ["Priority", "Hotel", "Destination", "Demand Level", "Searches", "Views", "Why prioritized", "Next step"]
        ],
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
