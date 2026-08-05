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

from dashboard.utils.data import load_engine, load_funnel
from dashboard.utils.ui import apply_theme, style_table


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
    for column in ["Previous_Searches", "Search_Change", "View_Change"]:
        if column not in detail:
            detail[column] = np.nan
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    if "Upload_Change_Status" not in detail:
        detail["Upload_Change_Status"] = "No baseline"
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


engine = load_engine()
funnel = load_funnel()
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
        '<div class="profile-card"><b>Market Manager</b></div>',
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

filtered_detail["Hotel Rising"] = filtered_detail["Search_Change"].gt(0)
filtered_detail["Hotel Action"] = (
    filtered_detail["Demand Level"].isin(["Very High", "High"])
    & filtered_detail["Search_Change"].gt(0)
)
filtered_daily = (
    filtered_detail.groupby("checkin_date", as_index=False)
    .agg(
        Latest_Searches=("search_volume", "sum"),
        Previous_Searches=("Previous_Searches", lambda values: values.sum(min_count=1)),
        Search_Change=("Search_Change", lambda values: values.sum(min_count=1)),
        Hotels=("ProductID", "nunique"),
        Hotels_Rising=("Hotel Rising", "sum"),
        Action_Hotels=("Hotel Action", "sum"),
    )
    .sort_values("checkin_date")
)
filtered_daily["Change_Pct"] = np.where(
    filtered_daily["Previous_Searches"].gt(0),
    filtered_daily["Search_Change"] / filtered_daily["Previous_Searches"],
    np.nan,
)
filtered_daily["Signal"] = np.select(
    [
        filtered_daily["Previous_Searches"].isna(),
        filtered_daily["Change_Pct"].ge(0.25),
        filtered_daily["Change_Pct"].ge(0.10),
        filtered_daily["Change_Pct"].le(-0.10),
    ],
    ["No baseline", "Critical surge", "High increase", "Declining"],
    default="Stable",
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
total_change = float(selected["Search_Change"].sum(min_count=1)) if selected["Search_Change"].notna().any() else np.nan
active_hotels = int(selected["ProductID"].nunique())
previous_total = float(selected["Previous_Searches"].sum(min_count=1)) if selected["Previous_Searches"].notna().any() else np.nan
change_pct = total_change / previous_total if previous_total and not pd.isna(total_change) else np.nan
rising_hotels = int(selected["Search_Change"].gt(0).sum())

hotel_median = selected["search_volume"].median() if not selected.empty else 0
hotel_q75 = selected["search_volume"].quantile(0.75) if not selected.empty else 0
selected["Change_Pct"] = np.where(
    selected["Previous_Searches"].gt(0),
    selected["Search_Change"] / selected["Previous_Searches"],
    np.nan,
)
selected["Signal"] = np.select(
    [
        selected["Previous_Searches"].isna(),
        selected["search_volume"].ge(hotel_q75) & selected["Change_Pct"].ge(0.25),
        selected["search_volume"].ge(hotel_median) & selected["Change_Pct"].ge(0.10),
        selected["Change_Pct"].le(-0.10),
    ],
    ["New demand", "Critical surge", "High increase", "Declining"],
    default="Stable",
)
action_hotels = int(
    (
        selected["Signal"].isin(["Critical surge", "High increase"])
        | (selected["Signal"].eq("New demand") & selected["search_volume"].ge(hotel_q75))
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
k1.metric("Demand signal", date_signal, help="Change in demand versus the previous upload")
k2.metric("Latest searches", f"{total_searches:,.0f}")
k3.metric(
    "Change vs previous upload",
    "No baseline" if pd.isna(change_pct) else f"{change_pct:+.1%}",
    delta=None if pd.isna(total_change) else f"{total_change:+,.0f} searches",
    help="Latest uploaded searches compared with the previous uploaded searches",
)
k4.metric("Hotels rising", f"{rising_hotels:,} of {active_hotels:,}")
k5.metric("Hotels requiring action", f"{action_hotels:,}")

chart_col, destination_col = st.columns([2.05, 1])
with chart_col:
    st.subheader("Demand change by check-in date")
    change_bars = (
        alt.Chart(filtered_daily)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("checkin_date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=-35)),
            y=alt.Y("Search_Change:Q", title="Search change", axis=alt.Axis(format="~s")),
            color=alt.condition(
                alt.datum.Search_Change >= 0,
                alt.value("#16a34a"),
                alt.value("#dc2626"),
            ),
            tooltip=[
                alt.Tooltip("checkin_date:T", title="Check-in", format="%d %b %Y"),
                alt.Tooltip("Latest_Searches:Q", title="Latest searches", format=",.0f"),
                alt.Tooltip("Search_Change:Q", title="Change", format="+,.0f"),
                alt.Tooltip("Change_Pct:Q", title="Change %", format="+.1%"),
                "Signal:N",
            ],
        )
        .properties(height=320)
    )
    rule = (
        alt.Chart(pd.DataFrame({"checkin_date": [selected_ts]}))
        .mark_rule(color="#f59e0b", strokeWidth=2, strokeDash=[5, 4])
        .encode(x="checkin_date:T")
    )
    st.altair_chart(change_bars + rule, width="stretch")

with destination_col:
    st.subheader("Demand by destination")
    destination_rank = (
        selected.groupby("Destination", as_index=False)
        .agg(Searches=("search_volume", "sum"), Views=("view_volume", "sum"), Hotels=("ProductID", "nunique"))
        .sort_values(["Searches", "Views"], ascending=False)
        .head(8)
    )
    st.dataframe(
        style_table(destination_rank),
        hide_index=True,
        width="stretch",
        height=354,
        column_config={
            "Searches": st.column_config.NumberColumn(format="localized"),
            "Views": st.column_config.NumberColumn(format="localized"),
        },
    )

st.subheader("Daily demand change")
daily_display = filtered_daily.rename(
    columns={
        "checkin_date": "Check-in Date",
        "Latest_Searches": "Latest Searches",
        "Change_Pct": "Change %",
        "Hotels_Rising": "Hotels Rising",
        "Action_Hotels": "Action Hotels",
    }
)
daily_display["Change %"] = daily_display["Change %"] * 100
st.dataframe(
    style_table(daily_display[
        ["Check-in Date", "Signal", "Latest Searches", "Change %", "Hotels Rising", "Action Hotels"]
    ]),
    hide_index=True,
    width="stretch",
    height=320,
    column_config={
        "Check-in Date": st.column_config.DateColumn(format="DD MMM YYYY"),
        "Latest Searches": st.column_config.NumberColumn(format="localized"),
        "Change %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Hotels Rising": st.column_config.NumberColumn(format="localized"),
        "Action Hotels": st.column_config.NumberColumn(format="localized"),
    },
)

st.subheader(f"Hotels to check — {selected_ts:%d %b %Y}")
st.caption(
    "Demand level compares this date with each hotel’s own available check-in dates. "
    "The dashboard does not show inventory or parity results."
)

signal_order = {"Critical surge": 5, "High increase": 4, "New demand": 3, "Stable": 2, "Declining": 1}
selected["Signal Order"] = selected["Signal"].map(signal_order).fillna(0)
selected = selected.sort_values(
    ["Signal Order", "search_volume", "Search_Change"],
    ascending=[False, False, False],
).reset_index(drop=True)
selected["Priority"] = np.arange(1, len(selected) + 1)


def next_step(row: pd.Series) -> str:
    if row["Signal"] == "Critical surge":
        return "Check inventory now → parity"
    if row["Signal"] == "High increase":
        return "Check inventory → parity"
    if row["Signal"] == "New demand":
        return "Validate demand → inventory"
    if row["Demand Level"] in ["Very High", "High"]:
        return "Check inventory → parity"
    if row["Signal"] == "Declining":
        return "Lower priority"
    return "Monitor"


selected["Next step"] = selected.apply(next_step, axis=1)

display = selected.rename(
    columns={
        "ProductName": "Hotel",
        "search_volume": "Latest",
        "Change_Pct": "Change %",
    }
)
display_columns = [
    "Priority",
    "Hotel",
    "Latest",
    "Change %",
    "Signal",
    "Next step",
]
display["Change %"] = display["Change %"] * 100
st.dataframe(
    style_table(display[display_columns].head(200)),
    hide_index=True,
    width="stretch",
    height=530,
    column_config={
        "Priority": st.column_config.NumberColumn(format="%d", width="small"),
        "Latest": st.column_config.NumberColumn(format="localized"),
        "Change %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Signal": st.column_config.TextColumn(width="medium"),
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
