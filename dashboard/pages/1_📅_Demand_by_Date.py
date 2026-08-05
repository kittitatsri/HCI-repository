from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data import load_demand, load_engine, load_funnel
from dashboard.utils.ui import apply_theme


st.set_page_config(page_title="HCI | Demand by Date", page_icon="📅", layout="wide")
apply_theme()


@st.cache_data(show_spinner=False)
def prepare_data(
    engine: pd.DataFrame, funnel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = engine[
        [
            "ProductID",
            "ProductName",
            "Destination",
            "Region",
            "HotelType Short",
            "Agoda Status",
            "Ctrip Status",
        ]
    ].drop_duplicates("ProductID")
    detail = funnel.drop(columns=["ProductName", "Destination"], errors="ignore").merge(
        master, on="ProductID", how="left", validate="many_to_one"
    )
    detail["checkin_date"] = pd.to_datetime(detail["checkin_date"], errors="coerce").dt.normalize()
    detail["search_volume"] = pd.to_numeric(detail["search_volume"], errors="coerce").fillna(0)
    detail["view_volume"] = pd.to_numeric(detail["view_volume"], errors="coerce").fillna(0)
    detail = detail.dropna(subset=["checkin_date"])

    hotel_baseline = (
        detail.groupby("ProductID")["search_volume"]
        .agg(
            Hotel_Search_Median="median",
            Hotel_Search_Q75=lambda values: values.quantile(0.75),
            Hotel_Search_Q90=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )
    detail = detail.merge(hotel_baseline, on="ProductID", how="left", validate="many_to_one")
    detail["Demand Level"] = np.select(
        [
            detail["search_volume"].ge(detail["Hotel_Search_Q90"]),
            detail["search_volume"].ge(detail["Hotel_Search_Q75"]),
            detail["search_volume"].ge(detail["Hotel_Search_Median"]),
        ],
        ["Very High", "High", "Medium"],
        default="Low",
    )
    daily = (
        detail.groupby("checkin_date", as_index=False)
        .agg(
            Searches=("search_volume", "sum"),
            Views=("view_volume", "sum"),
            Hotels=("ProductID", "nunique"),
        )
        .sort_values("checkin_date")
    )
    return master, detail, daily


@st.cache_data(show_spinner=False)
def latest_increase(demand: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    selected = demand[demand["checkin_date"].dt.normalize().eq(selected_date)].copy()
    selected["search_volume"] = pd.to_numeric(selected["search_volume"], errors="coerce").fillna(0)
    selected["view_volume"] = pd.to_numeric(selected["view_volume"], errors="coerce").fillna(0)
    selected = selected.sort_values(["ProductID", "snapshot_at"])
    selected["Previous_Searches"] = selected.groupby("ProductID")["search_volume"].shift(1)
    selected["Previous_Views"] = selected.groupby("ProductID")["view_volume"].shift(1)
    latest = selected.drop_duplicates("ProductID", keep="last").copy()
    latest["Search Increase"] = (latest["search_volume"] - latest["Previous_Searches"]).clip(lower=0)
    latest["View Increase"] = (latest["view_volume"] - latest["Previous_Views"]).clip(lower=0)
    return latest[["ProductID", "Search Increase", "View Increase"]]


engine = load_engine()
funnel = load_funnel()
demand = load_demand()
master, detail, portfolio_daily = prepare_data(engine, funnel)

with st.sidebar:
    st.markdown(
        '<div class="hci-brand"><b>HCI</b><span>HOTEL COMMERCIAL<br>INTELLIGENCE</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Demand-led hotel prioritization")

head_left, head_right = st.columns([4, 1])
with head_left:
    st.title("Demand by Date")
    st.caption("Select a check-in date, understand demand, and choose which hotels to check first.")
with head_right:
    st.markdown(
        '<div class="profile-card"><b>Kittitat Sri</b><br><span>Market Manager</span></div>',
        unsafe_allow_html=True,
    )

available_dates = portfolio_daily["checkin_date"].dt.date.tolist()
today = date.today()
default_date = today if today in available_dates else available_dates[0]

f1, f2, f3, f4 = st.columns([1.05, 1.35, 1.35, 1.35])
selected_date = f1.date_input(
    "Check-in date",
    value=default_date,
    min_value=min(available_dates),
    max_value=max(available_dates),
)
regions = f2.multiselect(
    "Region",
    sorted(master["Region"].dropna().astype(str).unique()),
    placeholder="All regions",
)
hotel_types = f3.multiselect(
    "Hotel Type",
    sorted(master["HotelType Short"].dropna().astype(str).unique()),
    placeholder="All hotel types",
)

eligible_master = master.copy()
if regions:
    eligible_master = eligible_master[eligible_master["Region"].isin(regions)]
if hotel_types:
    eligible_master = eligible_master[eligible_master["HotelType Short"].isin(hotel_types)]
destination_options = sorted(eligible_master["Destination"].dropna().astype(str).unique())
destinations = f4.multiselect("Destination", destination_options, placeholder="All destinations")

eligible_ids = set(eligible_master["ProductID"].dropna().astype(int))
filtered_detail = detail[detail["ProductID"].isin(eligible_ids)].copy()
if destinations:
    filtered_detail = filtered_detail[filtered_detail["Destination"].isin(destinations)]

selected_ts = pd.Timestamp(selected_date)
selected = filtered_detail[filtered_detail["checkin_date"].eq(selected_ts)].copy()
if "Search_Change" not in selected:
    selected["Search_Change"] = np.nan
if "View_Change" not in selected:
    selected["View_Change"] = np.nan
if "Upload_Change_Status" not in selected:
    selected["Upload_Change_Status"] = "No baseline"

filtered_daily = (
    filtered_detail.groupby("checkin_date", as_index=False)
    .agg(Searches=("search_volume", "sum"), Views=("view_volume", "sum"), Hotels=("ProductID", "nunique"))
    .sort_values("checkin_date")
)
portfolio_row = portfolio_daily[portfolio_daily["checkin_date"].eq(selected_ts)]
portfolio_searches = float(portfolio_row["Searches"].iloc[0]) if not portfolio_row.empty else 0
portfolio_quantiles = portfolio_daily["Searches"].quantile([0.40, 0.75, 0.90])
date_level = (
    "Very High"
    if portfolio_searches >= portfolio_quantiles.loc[0.90]
    else "High"
    if portfolio_searches >= portfolio_quantiles.loc[0.75]
    else "Medium"
    if portfolio_searches >= portfolio_quantiles.loc[0.40]
    else "Low"
)

total_searches = float(selected["search_volume"].sum())
total_views = float(selected["view_volume"].sum())
total_change = float(selected["Search_Change"].sum(min_count=1)) if selected["Search_Change"].notna().any() else np.nan
active_hotels = int(selected["ProductID"].nunique())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Portfolio demand", date_level, help="Selected date compared with other check-in dates")
k2.metric("Searches", f"{total_searches:,.0f}")
k3.metric("Views", f"{total_views:,.0f}")
k4.metric(
    "Search change vs previous upload",
    "No baseline" if pd.isna(total_change) else f"{total_change:+,.0f}",
    help="Latest uploaded demand minus the previous uploaded demand for the selected check-in date",
)
k5.metric("Hotels with demand", f"{active_hotels:,}")

chart_col, destination_col = st.columns([2.05, 1])
with chart_col:
    st.subheader("Demand across check-in dates")
    chart_data = filtered_daily.melt(
        id_vars="checkin_date",
        value_vars=["Searches", "Views"],
        var_name="Metric",
        value_name="Volume",
    )
    lines = (
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
        .properties(height=320)
    )
    rule = (
        alt.Chart(pd.DataFrame({"checkin_date": [selected_ts]}))
        .mark_rule(color="#f59e0b", strokeWidth=2, strokeDash=[5, 4])
        .encode(x="checkin_date:T")
    )
    st.altair_chart(lines + rule, width="stretch")

with destination_col:
    st.subheader("Demand by destination")
    destination_rank = (
        selected.groupby("Destination", as_index=False)
        .agg(Searches=("search_volume", "sum"), Views=("view_volume", "sum"), Hotels=("ProductID", "nunique"))
        .sort_values(["Searches", "Views"], ascending=False)
        .head(8)
    )
    st.dataframe(
        destination_rank,
        hide_index=True,
        width="stretch",
        height=354,
        column_config={
            "Searches": st.column_config.NumberColumn(format="localized"),
            "Views": st.column_config.NumberColumn(format="localized"),
        },
    )

st.subheader(f"Hotels to check — {selected_ts:%d %b %Y}")
st.caption(
    "Demand level compares this date with each hotel’s own available check-in dates. "
    "The dashboard does not show inventory or parity results."
)

level_order = {"Very High": 4, "High": 3, "Medium": 2, "Low": 1}
selected["Level Order"] = selected["Demand Level"].map(level_order).fillna(0)
selected = selected.sort_values(
    ["Level Order", "search_volume", "Search_Change", "view_volume"],
    ascending=[False, False, False, False],
).reset_index(drop=True)
selected["Priority"] = np.arange(1, len(selected) + 1)
selected["Why prioritized"] = np.select(
    [
        selected["Demand Level"].eq("Very High") & selected["Search_Change"].gt(0),
        selected["Demand Level"].eq("Very High"),
        selected["Demand Level"].eq("High") & selected["Search_Change"].gt(0),
        selected["Demand Level"].eq("High"),
    ],
    ["Very high and rising", "Very high demand", "High and rising", "High demand"],
    default="Demand to monitor",
)
mapping_labels = {
    "Mapped": "Mapped",
    "No Room": "Room mapping needed",
    "Not Live": "Not live",
    "Not Mapped": "Not mapped",
    "Unknown": "Check mapping",
}
selected["Agoda Mapping"] = selected["Agoda Status"].map(mapping_labels).fillna("Check mapping")
selected["Ctrip Mapping"] = selected["Ctrip Status"].map(mapping_labels).fillna("Check mapping")


def next_step(row: pd.Series) -> str:
    actions: list[str] = []
    channel_actions = {
        "Not Mapped": "map {channel}",
        "No Room": "map {channel} rooms",
        "Not Live": "activate {channel}",
        "Unknown": "verify {channel}",
    }
    for channel, status_column in (("Agoda", "Agoda Status"), ("Ctrip", "Ctrip Status")):
        template = channel_actions.get(str(row[status_column]))
        if template:
            actions.append(template.format(channel=channel))
    base = "Check inventory → parity"
    return base if not actions else base + " → " + " + ".join(actions)


selected["Next step"] = selected.apply(next_step, axis=1)

display = selected.rename(
    columns={
        "ProductName": "Hotel",
        "HotelType Short": "Hotel Type",
        "search_volume": "Searches",
        "view_volume": "Views",
        "Previous_Searches": "Previous Searches",
        "Search_Change": "Search Change",
        "Upload_Change_Status": "Change",
    }
)
display_columns = [
    "Priority",
    "Hotel",
    "Destination",
    "Region",
    "Hotel Type",
    "Demand Level",
    "Searches",
    "Views",
    "Previous Searches",
    "Search Change",
    "Change",
    "Why prioritized",
    "Agoda Mapping",
    "Ctrip Mapping",
    "Next step",
]
st.dataframe(
    display[display_columns].head(200),
    hide_index=True,
    width="stretch",
    height=530,
    column_config={
        "Priority": st.column_config.NumberColumn(format="%d", width="small"),
        "Searches": st.column_config.NumberColumn(format="localized"),
        "Views": st.column_config.NumberColumn(format="localized"),
        "Previous Searches": st.column_config.NumberColumn(format="localized"),
        "Search Change": st.column_config.NumberColumn(format="localized"),
        "Agoda Mapping": st.column_config.TextColumn(width="medium"),
        "Ctrip Mapping": st.column_config.TextColumn(width="medium"),
        "Next step": st.column_config.TextColumn(width="medium"),
    },
)

download_columns = ["ProductID", "checkin_date"] + display_columns
st.download_button(
    "Download filtered hotel priorities",
    display[download_columns].to_csv(index=False).encode("utf-8-sig"),
    file_name=f"hotel_demand_priorities_{selected_ts:%Y-%m-%d}.csv",
    mime="text/csv",
)

st.markdown(
    '<div class="workflow-note"><b>Next:</b> Check inventory manually for the highest-demand hotels, '
    'then open the existing rate-parity tool.</div>',
    unsafe_allow_html=True,
)
