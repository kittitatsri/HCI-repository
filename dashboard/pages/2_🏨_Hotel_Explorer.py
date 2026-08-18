from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data import build_hotel_checkin_trend, load_demand, load_engine, load_funnel
from dashboard.utils.ui import apply_theme, style_table


st.set_page_config(page_title="HCI | Hotel Explorer", page_icon="🏨", layout="wide")
apply_theme()


@st.cache_data(show_spinner=False)
def prepare_hotel_data(
    engine: pd.DataFrame, funnel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_columns = [
        "ProductID",
        "ProductName",
        "Destination",
        "Region",
        "HotelType Short",
        "Agoda Status",
        "Ctrip Status",
    ]
    master = engine[master_columns].drop_duplicates("ProductID").copy()
    master["ProductID"] = pd.to_numeric(master["ProductID"], errors="coerce").astype("Int64")
    detail = funnel.merge(
        master[["ProductID", "Region", "HotelType Short", "Agoda Status", "Ctrip Status"]],
        on="ProductID",
        how="left",
        validate="many_to_one",
    )
    detail["checkin_date"] = pd.to_datetime(detail["checkin_date"], errors="coerce").dt.normalize()
    for column in [
        "Destination_Searches",
        "Previous_Destination_Searches",
        "Destination_Search_Change",
        "Previous_Views",
        "View_Change",
    ]:
        if column not in detail:
            detail[column] = np.nan
    for column in [
        "view_volume",
        "Destination_Searches",
        "Previous_Destination_Searches",
        "Destination_Search_Change",
        "Previous_Views",
        "View_Change",
        "Internal_Worst_Gap",
        "Internal_Median_Gap",
        "Internal_Comparable_Rates",
        "Internal_Total_Rates",
        "Agoda_Price_Disadvantage",
    ]:
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    return master, detail.dropna(subset=["checkin_date"])


def internal_status(gap: float | None) -> tuple[str, str, str]:
    if pd.isna(gap):
        return "No comparison", "No comparable partner rate", "neutral"
    if gap <= -0.03:
        return "Worse", f"{abs(gap):.1%} more expensive", "danger"
    if gap >= 0.03:
        return "Cheaper", f"{gap:.1%} cheaper", "success"
    return "Competitive", f"Within {abs(gap):.1%}", "info"


def agoda_status(disadvantage: float | None) -> tuple[str, str, str]:
    if pd.isna(disadvantage):
        return "No comparison", "No Agoda comparison", "neutral"
    if disadvantage == 0:
        return "Cheapest", "0% disadvantage", "success"
    if disadvantage <= 0.03:
        return "Competitive", f"{disadvantage:.1%} disadvantage", "info"
    return "Disadvantaged", f"{disadvantage:.1%} disadvantage", "danger"


def display_mapping(status: object) -> tuple[str, str]:
    labels = {
        "Mapped": ("Mapped", "success"),
        "No Room": ("Room mapping needed", "warning"),
        "Not Live": ("Not live", "warning"),
        "Not Mapped": ("Not mapped", "danger"),
        "Unknown": ("Check mapping", "neutral"),
    }
    return labels.get(str(status), ("Check mapping", "neutral"))


def tool_card(title: str, status: str, detail: str, tone: str) -> None:
    st.markdown(
        f'<div class="tool-card"><div class="tool-title">{title}</div>'
        f'<span class="tool-status {tone}">{status}</span>'
        f'<div class="tool-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


engine = load_engine()
funnel = load_funnel()
master, detail = prepare_hotel_data(engine, funnel)

with st.sidebar:
    st.markdown(
        '<div class="hci-brand"><b>HCI</b><span>HOTEL COMMERCIAL<br>INTELLIGENCE</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Demand-led hotel prioritization")

head_left, head_right = st.columns([4, 1])
with head_left:
    st.title("Hotel Explorer")
    st.caption("Investigate one hotel for one check-in date.")
with head_right:
    st.markdown(
        '<div class="profile-card"><b>Market Manager</b></div>',
        unsafe_allow_html=True,
    )

query_product = st.query_params.get("product_id")
query_date = st.query_params.get("checkin_date")
try:
    query_product_id = int(query_product) if query_product else None
except (TypeError, ValueError):
    query_product_id = None

f1, f2 = st.columns(2)
regions = f1.multiselect(
    "Region",
    sorted(master["Region"].dropna().astype(str).unique()),
    placeholder="All regions",
)
hotel_types = f2.multiselect(
    "Hotel Type",
    sorted(master["HotelType Short"].dropna().astype(str).unique()),
    placeholder="All hotel types",
)

eligible = master.copy()
if regions:
    eligible = eligible[eligible["Region"].isin(regions)]
if hotel_types:
    eligible = eligible[eligible["HotelType Short"].isin(hotel_types)]
eligible = eligible.sort_values(["ProductName", "ProductID"])
eligible["Hotel Label"] = eligible.apply(
    lambda row: f"{row['ProductName']} · {int(row['ProductID'])}", axis=1
)

default_hotel_index = 0
if query_product_id in set(eligible["ProductID"].dropna().astype(int)):
    default_hotel_index = int(
        np.flatnonzero(eligible["ProductID"].astype(int).eq(query_product_id).to_numpy())[0]
    )
selected_label = st.selectbox(
    "Hotel",
    eligible["Hotel Label"].tolist(),
    index=default_hotel_index,
    placeholder="Search by hotel name or Product ID",
)
product_id = int(selected_label.rsplit(" · ", 1)[1])
hotel = master[master["ProductID"].eq(product_id)].iloc[0]
hotel_detail = detail[detail["ProductID"].eq(product_id)].sort_values("checkin_date").copy()

available_dates = hotel_detail["checkin_date"].dt.date.tolist()
if not available_dates:
    st.warning("This hotel has no demand records in the current dataset.")
    st.stop()
try:
    requested_date = pd.Timestamp(query_date).date() if query_date else None
except (TypeError, ValueError):
    requested_date = None
today = date.today()
default_date = (
    requested_date
    if requested_date in available_dates
    else today
    if today in available_dates
    else available_dates[0]
)
selected_date = st.date_input(
    "Check-in Date",
    value=default_date,
    min_value=min(available_dates),
    max_value=max(available_dates),
)
selected_ts = pd.Timestamp(selected_date)
selected_rows = hotel_detail[hotel_detail["checkin_date"].eq(selected_ts)]
if selected_rows.empty:
    st.info("No demand record exists for this hotel on the selected check-in date.")
    st.stop()
selected = selected_rows.iloc[0]

view_median = hotel_detail["view_volume"].median()
view_q75 = hotel_detail["view_volume"].quantile(0.75)
view_q90 = hotel_detail["view_volume"].quantile(0.90)
selected_views = float(selected["view_volume"])
demand_level = (
    "Very High"
    if selected_views >= view_q90
    else "High"
    if selected_views >= view_q75
    else "Medium"
    if selected_views >= view_median
    else "Low"
)
baseline_change = selected_views / view_median - 1 if view_median else np.nan
hotel_detail["View Change %"] = np.where(
    hotel_detail["Previous_Views"].gt(0),
    hotel_detail["View_Change"] / hotel_detail["Previous_Views"],
    np.nan,
)
hotel_detail["Hotel Signal"] = np.select(
    [
        hotel_detail["Previous_Views"].isna() | hotel_detail["check_status"].eq("new entry"),
        hotel_detail["View Change %"].ge(0.25),
        hotel_detail["View Change %"].ge(0.10),
        hotel_detail["View Change %"].le(-0.10),
    ],
    ["New interest", "Critical surge", "High increase", "Declining"],
    default="Stable",
)
destination_change_pct = np.where(
    hotel_detail["Previous_Destination_Searches"].gt(0),
    hotel_detail["Destination_Search_Change"] / hotel_detail["Previous_Destination_Searches"],
    np.nan,
)
hotel_detail["Destination Signal"] = np.select(
    [
        hotel_detail["Previous_Destination_Searches"].isna(),
        pd.Series(destination_change_pct, index=hotel_detail.index).ge(0.25),
        pd.Series(destination_change_pct, index=hotel_detail.index).ge(0.10),
        pd.Series(destination_change_pct, index=hotel_detail.index).le(-0.10),
    ],
    ["New demand", "Critical surge", "High increase", "Declining"],
    default="Stable",
)
selected = hotel_detail[hotel_detail["checkin_date"].eq(selected_ts)].iloc[0]
selected_change = float(selected["View_Change"]) if pd.notna(selected["View_Change"]) else np.nan
selected_change_pct = (
    selected_change / float(selected["Previous_Views"])
    if pd.notna(selected_change) and pd.notna(selected["Previous_Views"]) and selected["Previous_Views"] > 0
    else np.nan
)
upload_change_text = "No baseline" if pd.isna(selected_change) else f"{selected_change:+,.0f} hotel views"
change_detail_text = (
    "no baseline" if pd.isna(selected_change_pct) else f"{selected_change_pct:+.1%} vs previous"
)

st.markdown(f"### {hotel['ProductName']}")
st.caption(
    f"Product ID {product_id} · {hotel['Destination']} · {hotel['Region']} · "
    f"Hotel Type {hotel['HotelType Short']}"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Destination demand", selected["Destination Signal"], help="Destination search change")
k2.metric("Destination searches", f"{selected['Destination_Searches']:,.0f}")
k3.metric("Hotel views", f"{selected_views:,.0f}")
k4.metric(
    "Hotel view change",
    "No baseline" if pd.isna(selected_change_pct) else f"{selected_change_pct:+.1%}",
    delta=None if pd.isna(selected_change) else f"{selected_change:+,.0f} views",
)

chart_col, why_col = st.columns([2.15, 1])
with chart_col:
    st.subheader("Demand by Check-in Date")
    chart_data = build_hotel_checkin_trend(load_demand(), product_id)
    lines = (
        alt.Chart(chart_data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("checkin_date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=-35)),
            y=alt.Y("Hotel Views:Q", title="Observed hotel views", axis=alt.Axis(format="~s")),
            color=alt.value("#16a34a"),
            tooltip=[
                alt.Tooltip("checkin_date:T", title="Check-in", format="%d %b %Y"),
                alt.Tooltip("Hotel Views:Q", title="Observed hotel views", format=",.0f"),
            ],
        )
        .properties(height=315)
    )
    selected_rule = (
        alt.Chart(pd.DataFrame({"checkin_date": [selected_ts]}))
        .mark_rule(color="#f59e0b", strokeWidth=2, strokeDash=[5, 4])
        .encode(x="checkin_date:T")
    )
    st.altair_chart(lines + selected_rule, width="stretch")
    st.caption("Uses all stored observation intervals. Destination searches are excluded because they are destination-level, not hotel-level.")

with why_col:
    st.subheader("Why prioritized")
    relative_text = (
        f"{baseline_change:.0%} above recent baseline"
        if pd.notna(baseline_change) and baseline_change >= 0
        else f"{abs(baseline_change):.0%} below recent baseline"
        if pd.notna(baseline_change)
        else "Baseline unavailable"
    )
    peak_date = hotel_detail.loc[hotel_detail["view_volume"].idxmax(), "checkin_date"]
    st.markdown(
        f'<div class="fact-list"><div><b>Demand:</b> {relative_text}</div>'
        f'<div><b>Upload change:</b> {upload_change_text}</div>'
        f'<div><b>Peak check-in:</b> {peak_date:%d %b %Y}</div></div>',
        unsafe_allow_html=True,
    )

internal_label, internal_value, internal_tone = internal_status(selected["Internal_Worst_Gap"])
agoda_label, agoda_value, agoda_tone = agoda_status(selected["Agoda_Price_Disadvantage"])
agoda_mapping, agoda_mapping_tone = display_mapping(hotel["Agoda Status"])
ctrip_mapping, ctrip_mapping_tone = display_mapping(hotel["Ctrip Status"])
rate_issue = internal_tone == "danger" or agoda_tone == "danger"
rate_missing = internal_tone == "neutral" and agoda_tone == "neutral"
distribution_issue = agoda_mapping_tone in {"danger", "warning", "neutral"} or ctrip_mapping_tone in {
    "danger",
    "warning",
    "neutral",
}

st.subheader("Issue Summary")
t1, t2, t3, t4 = st.columns(4)
with t1:
    tool_card("Inventory", "Manual check", "Not available in HCI", "neutral")
with t2:
    rate_summary = f"Internal: {internal_value} · Agoda: {agoda_value}"
    tool_card(
        "Rate Competitiveness",
        "Issue found" if rate_issue else "No comparison" if rate_missing else "No major issue",
        rate_summary,
        "danger" if rate_issue else "neutral" if rate_missing else "success",
    )
with t3:
    tool_card(
        "Distribution",
        "Action needed" if distribution_issue else "Healthy",
        f"Agoda: {agoda_mapping} · Ctrip: {ctrip_mapping}",
        "warning" if distribution_issue else "success",
    )
with t4:
    tool_card(
        "Demand Detail",
        demand_level,
        f"{selected_views:,.0f} hotel views · {change_detail_text}",
        "danger" if demand_level == "Very High" else "warning" if demand_level == "High" else "info",
    )

workflow_steps = ["Check inventory manually"]
if rate_issue:
    workflow_steps.append("open Rate Competitiveness and verify room/rate conditions")
elif rate_missing:
    workflow_steps.append("check parity in the external tool")
if agoda_mapping != "Mapped":
    workflow_steps.append(f"resolve Agoda: {agoda_mapping.lower()}")
if ctrip_mapping != "Mapped":
    workflow_steps.append(f"resolve Ctrip: {ctrip_mapping.lower()}")
workflow = " → ".join(workflow_steps)
st.markdown(
    f'<div class="workflow-note"><b>Recommended workflow:</b> {workflow}.</div>',
    unsafe_allow_html=True,
)

demand_tab, parity_tab, distribution_tab = st.tabs(
    ["Demand Detail", "Rate Competitiveness", "Distribution"]
)

with demand_tab:
    st.subheader("Daily Demand Detail")
    demand_display = hotel_detail[
        [
            "checkin_date",
            "Destination_Searches",
            "Destination Signal",
            "view_volume",
            "View Change %",
            "Hotel Signal",
        ]
    ].copy()
    demand_display["Hotel Interest"] = np.select(
        [
            demand_display["view_volume"].ge(view_q90),
            demand_display["view_volume"].ge(view_q75),
            demand_display["view_volume"].ge(view_median),
        ],
        ["Very High", "High", "Medium"],
        default="Low",
    )
    demand_display = demand_display.rename(
        columns={
            "checkin_date": "Check-in Date",
            "Destination_Searches": "Destination Searches",
            "view_volume": "Hotel Views",
        }
    )
    demand_display["View Change %"] = demand_display["View Change %"] * 100
    st.dataframe(
        style_table(demand_display[
            [
                "Check-in Date",
                "Destination Signal",
                "Destination Searches",
                "Hotel Interest",
                "Hotel Views",
                "View Change %",
                "Hotel Signal",
            ]
        ]),
        hide_index=True,
        width="stretch",
        height=330,
        column_config={
            "Check-in Date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Destination Searches": st.column_config.NumberColumn(format="localized"),
            "Hotel Views": st.column_config.NumberColumn(format="localized"),
            "View Change %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Destination Signal": st.column_config.TextColumn(width="medium"),
            "Hotel Signal": st.column_config.TextColumn(width="medium"),
        },
    )

with parity_tab:
    st.subheader(f"Rate Competitiveness · {selected_ts:%d %b %Y}")
    st.caption(
        "Hotel/date parity signal — verify room type, occupancy, meal plan, cancellation policy, taxes, "
        "and currency before contacting the hotel."
    )
    p1, p2 = st.columns(2)
    with p1:
        with st.container(border=True):
            st.markdown("#### Internal Partner Parity")
            st.caption("Our rate compared with our partner rate")
            st.markdown(f"**{internal_label} · {internal_value}**")
            st.metric(
                "Median position",
                f"{selected['Internal_Median_Gap']:.1%}"
                if pd.notna(selected["Internal_Median_Gap"])
                else "No comparison",
            )
            comparable = int(selected["Internal_Comparable_Rates"] or 0) if pd.notna(selected["Internal_Comparable_Rates"]) else 0
            total = int(selected["Internal_Total_Rates"] or 0) if pd.notna(selected["Internal_Total_Rates"]) else 0
            st.write(f"Comparable rates: **{comparable} of {total}**")
    with p2:
        with st.container(border=True):
            st.markdown("#### Agoda Market Parity")
            st.caption("Our rate compared with Agoda’s market comparison")
            st.markdown(f"**{agoda_label} · {agoda_value}**")
            st.metric(
                "Price disadvantage",
                f"{selected['Agoda_Price_Disadvantage']:.1%}"
                if pd.notna(selected["Agoda_Price_Disadvantage"])
                else "No comparison",
            )
            st.write(f"Comparison date: **{selected_ts:%d %b %Y}**")

    if rate_issue:
        st.error(
            f"Demand is {demand_level.lower()}. Internal parity is {internal_label.lower()} and Agoda is "
            f"{agoda_label.lower()}. Verify conditions, then review the partner rate, markup, and hotel offer."
        )
    elif rate_missing:
        st.info("No parity comparison is available for this hotel and date. Check the external parity tool manually.")
    else:
        st.success("No major rate disadvantage is visible for the selected hotel and date.")

    parity_history = hotel_detail[
        [
            "checkin_date",
            "Internal_Worst_Gap",
            "Internal_Median_Gap",
            "Internal_Comparable_Rates",
            "Internal_Total_Rates",
            "Agoda_Price_Disadvantage",
        ]
    ].copy()
    parity_history = parity_history[
        parity_history["Internal_Worst_Gap"].notna()
        | parity_history["Agoda_Price_Disadvantage"].notna()
    ]
    with st.expander("View parity history", expanded=False):
        st.dataframe(style_table(parity_history), hide_index=True, width="stretch")

    parity_tool_url = os.getenv("PARITY_TOOL_URL", "").strip()
    if parity_tool_url:
        st.link_button("Open rate-parity tool", parity_tool_url, type="primary")
    else:
        st.button(
            "Open rate-parity tool",
            disabled=True,
            help="Set PARITY_TOOL_URL to enable this link.",
        )

with distribution_tab:
    st.subheader("Distribution")
    d1, d2 = st.columns(2)
    d1.metric("Agoda", agoda_mapping)
    d2.metric("Ctrip", ctrip_mapping)
    distribution_actions = []
    if agoda_mapping != "Mapped":
        distribution_actions.append(f"Agoda: {agoda_mapping}")
    if ctrip_mapping != "Mapped":
        distribution_actions.append(f"Ctrip: {ctrip_mapping}")
    if distribution_actions:
        st.warning("Action required — " + " · ".join(distribution_actions))
    else:
        st.success("The hotel is mapped on both Agoda and Ctrip.")

st.caption(
    f"Demand data updated {hotel_detail['snapshot_at'].max():%d %b %Y, %H:%M}. "
    "Inventory results are not stored in HCI."
)
