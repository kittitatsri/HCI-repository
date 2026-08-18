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

from dashboard.utils.data import (
    build_weekly_comparison,
    iso_week_options,
    load_demand,
    load_engine,
    load_funnel,
)
from dashboard.utils.checkin_week import build_checkin_week_comparison
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
    detail = funnel.merge(
        master[
            [
                "ProductID",
                "Region",
                "HotelType Short",
                "Agoda Status",
                "Ctrip Status",
            ]
        ],
        on="ProductID",
        how="left",
        validate="many_to_one",
    )
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
    if "Upload_Change_Status" not in detail:
        detail["Upload_Change_Status"] = "No baseline"
    detail = detail.dropna(subset=["checkin_date"])

    hotel_baseline = (
        detail.groupby("ProductID")["view_volume"]
        .agg(
            Hotel_View_Median="median",
            Hotel_View_Q75=lambda values: values.quantile(0.75),
            Hotel_View_Q90=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )
    detail = detail.merge(hotel_baseline, on="ProductID", how="left", validate="many_to_one")
    detail["Demand Level"] = np.select(
        [
            detail["view_volume"].ge(detail["Hotel_View_Q90"]),
            detail["view_volume"].ge(detail["Hotel_View_Q75"]),
            detail["view_volume"].ge(detail["Hotel_View_Median"]),
        ],
        ["Very High", "High", "Medium"],
        default="Low",
    )
    destination = detail.drop_duplicates(["Destination", "checkin_date"])
    searches = destination.groupby("checkin_date", as_index=False).agg(
        Searches=("Destination_Searches", "sum")
    )
    hotels = detail.groupby("checkin_date", as_index=False).agg(
        Views=("view_volume", "sum"), Hotels=("ProductID", "nunique")
    )
    daily = searches.merge(hotels, on="checkin_date", how="outer").sort_values("checkin_date")
    return master, detail, daily


def render_checkin_week(
    master: pd.DataFrame, detail: pd.DataFrame, week_start: pd.Timestamp
) -> None:
    f1, f2, f3 = st.columns(3)
    regions = f1.multiselect(
        "Region", sorted(master["Region"].dropna().astype(str).unique()), placeholder="All regions"
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
    filtered = detail[detail["ProductID"].isin(eligible["ProductID"])].copy()
    destinations = f3.multiselect(
        "Destination",
        sorted(filtered["Destination"].dropna().astype(str).unique()),
        placeholder="All destinations",
    )
    if destinations:
        filtered = filtered[filtered["Destination"].isin(destinations)]

    hotels, destination_week, weekday = build_checkin_week_comparison(filtered, week_start)
    week_end = week_start + pd.Timedelta(days=6)
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
    definition = (
        "Is demand for this check-in week higher or lower than demand for last check-in week? "
        "Compares check-in dates in the selected ISO week with dates in the previous ISO week."
    )
    hotels["Destination Signal"] = np.select(
        [
            hotels["Previous_Searches"].eq(0) & hotels["Current_Searches"].gt(0),
            hotels["Change_Pct"].ge(0.25), hotels["Change_Pct"].ge(0.10),
            hotels["Change_Pct"].le(-0.10),
        ],
        ["New demand", "Critical surge", "High increase", "Declining"],
        default="Stable",
    )
    q75 = hotels["Current_Views"].quantile(0.75) if not hotels.empty else 0
    hotels["Action"] = (
        hotels["Destination Signal"].isin(["Critical surge", "High increase", "New demand"])
        & (hotels["View_Change"].gt(0) | hotels["Current_Views"].ge(q75))
    )
    st.caption(f"Check-in week: {week_start:%d %b}–{week_end:%d %b %Y}")
    if not complete:
        st.warning(
            f"Incomplete comparison: {current_days}/7 selected-week dates and {previous_days}/7 "
            "previous-week dates. Change uses average searches per available day."
        )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Check-in week demand signal", signal, help=definition)
    k2.metric("Selected week searches", f"{current_total:,.0f}", help=definition)
    k3.metric(
        "Change vs previous check-in week",
        "No baseline" if pd.isna(change_pct) else f"{change_pct:+.1%}",
        delta=None if pd.isna(change) else f"{change:+,.0f} {'searches/day' if not complete else 'searches'}",
        help=definition,
    )
    k4.metric("Hotels with higher weekly views", f"{int(hotels['View_Change'].gt(0).sum()):,} of {len(hotels):,}")
    k5.metric("Hotels requiring action", f"{int(hotels['Action'].sum()):,}")

    st.subheader("This check-in week vs previous week", help=definition)
    chart = (
        alt.Chart(weekday)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("Weekday:O", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], title=None),
            y=alt.Y("Searches:Q", title="Destination searches", axis=alt.Axis(format="~s")),
            color=alt.Color(
                "Period:N",
                scale=alt.Scale(domain=["Selected week", "Previous week"], range=["#2563eb", "#94a3b8"]),
                title=None,
            ),
            tooltip=["Period:N", "Weekday:N", alt.Tooltip("Searches:Q", format=",.0f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")

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
        columns={"ProductName": "Hotel", "Current_Views": "This Week Views", "View_Change_Pct": "View Change %"}
    )
    display["View Change %"] = display["View Change %"] * 100
    st.dataframe(
        style_table(display[[
            "Priority", "Hotel", "Destination", "This Week Views",
            "View Change %", "Work Priority",
        ]].head(200)),
        hide_index=True,
        width="stretch",
        height=530,
        column_config={
            "Priority": st.column_config.NumberColumn(format="%d", width="small"),
            "This Week Views": st.column_config.NumberColumn(format="localized"),
            "View Change %": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )


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
    master, detail, portfolio_daily = prepare_data(engine, weekly_funnel)
    comparison_label = "previous ISO week"
elif view_mode == "Check-in Week":
    week_starts = (
        portfolio_daily["checkin_date"]
        - pd.to_timedelta(portfolio_daily["checkin_date"].dt.weekday, unit="D")
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
    render_checkin_week(master, detail, pd.Timestamp(week_options[selected_label]))
    st.stop()
else:
    week_col.caption(
        "Daily compares summed intervals from the newest upload with an equally sized preceding window."
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
destination_options = sorted(
    detail.loc[detail["ProductID"].isin(eligible_master["ProductID"]), "Destination"]
    .dropna()
    .astype(str)
    .unique()
)
destinations = f4.multiselect("Destination", destination_options, placeholder="All destinations")

eligible_ids = set(eligible_master["ProductID"].dropna().astype(int))
filtered_detail = detail[detail["ProductID"].isin(eligible_ids)].copy()
if destinations:
    filtered_detail = filtered_detail[filtered_detail["Destination"].isin(destinations)]

selected_ts = pd.Timestamp(selected_date)
destination_detail = filtered_detail.drop_duplicates(["Destination", "checkin_date"]).copy()
destination_detail["Change_Pct"] = np.where(
    destination_detail["Previous_Destination_Searches"].gt(0),
    destination_detail["Destination_Search_Change"]
    / destination_detail["Previous_Destination_Searches"],
    np.nan,
)
destination_detail["Destination Signal"] = np.select(
    [
        destination_detail["Previous_Destination_Searches"].isna(),
        destination_detail["Change_Pct"].ge(0.25),
        destination_detail["Change_Pct"].ge(0.10),
        destination_detail["Change_Pct"].le(-0.10),
    ],
    ["New demand", "Critical surge", "High increase", "Declining"],
    default="Stable",
)
filtered_detail = filtered_detail.merge(
    destination_detail[["Destination", "checkin_date", "Destination Signal"]],
    on=["Destination", "checkin_date"],
    how="left",
    validate="many_to_one",
)
filtered_detail["Hotel Rising"] = filtered_detail["View_Change"].gt(0)
filtered_detail["Strong Hotel Views"] = filtered_detail["view_volume"].ge(
    filtered_detail.groupby("checkin_date")["view_volume"].transform(lambda values: values.quantile(0.75))
)
filtered_detail["Hotel Action"] = (
    filtered_detail["Destination Signal"].isin(["Critical surge", "High increase", "New demand"])
    & (filtered_detail["Hotel Rising"] | filtered_detail["Strong Hotel Views"])
)
daily_search = destination_detail.groupby("checkin_date", as_index=False).agg(
    Latest_Searches=("Destination_Searches", "sum"),
    Previous_Searches=("Previous_Destination_Searches", lambda values: values.sum(min_count=1)),
    Search_Change=("Destination_Search_Change", lambda values: values.sum(min_count=1)),
)
daily_hotels = filtered_detail.groupby("checkin_date", as_index=False).agg(
    Hotels=("ProductID", "nunique"),
    Hotels_Rising=("Hotel Rising", "sum"),
    Action_Hotels=("Hotel Action", "sum"),
)
filtered_daily = daily_search.merge(daily_hotels, on="checkin_date", how="outer").sort_values(
    "checkin_date"
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

selected = filtered_detail[filtered_detail["checkin_date"].eq(selected_ts)].copy()
selected_destinations = destination_detail[destination_detail["checkin_date"].eq(selected_ts)]
total_searches = float(selected_destinations["Destination_Searches"].sum())
total_change = (
    float(selected_destinations["Destination_Search_Change"].sum(min_count=1))
    if selected_destinations["Destination_Search_Change"].notna().any()
    else np.nan
)
active_hotels = int(selected["ProductID"].nunique())
previous_total = (
    float(selected_destinations["Previous_Destination_Searches"].sum(min_count=1))
    if selected_destinations["Previous_Destination_Searches"].notna().any()
    else np.nan
)
change_pct = total_change / previous_total if previous_total and not pd.isna(total_change) else np.nan
rising_hotels = int(selected["View_Change"].gt(0).sum())
selected["View Change %"] = np.where(
    selected["Previous_Views"].gt(0),
    selected["View_Change"] / selected["Previous_Views"],
    np.nan,
)
selected["Hotel Signal"] = np.select(
    [
        selected["Previous_Views"].isna() | selected["check_status"].eq("new entry"),
        selected["View Change %"].ge(0.25),
        selected["View Change %"].ge(0.10),
        selected["View Change %"].le(-0.10),
    ],
    ["New interest", "Critical surge", "High increase", "Declining"],
    default="Stable",
)
action_hotels = int(selected["Hotel Action"].sum())
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
k2.metric("Destination searches", f"{total_searches:,.0f}")
k3.metric(
    "Change vs previous snapshot week" if view_mode == "Snapshot Week" else "Change vs previous upload",
    "No baseline" if pd.isna(change_pct) else f"{change_pct:+.1%}",
    delta=None if pd.isna(total_change) else f"{total_change:+,.0f} searches",
    help=f"Latest observed searches compared with the {comparison_label}",
)
k4.metric("Hotels with rising views", f"{rising_hotels:,} of {active_hotels:,}")
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
        .agg(
            Searches=("Destination_Searches", "first"),
            Previous_Searches=("Previous_Destination_Searches", "first"),
            Views=("view_volume", "sum"),
            Hotels=("ProductID", "nunique"),
        )
        .sort_values(["Searches", "Views"], ascending=False)
        .head(8)
    )
    destination_rank["Search Change %"] = np.where(
        destination_rank["Previous_Searches"].gt(0),
        (destination_rank["Searches"] - destination_rank["Previous_Searches"])
        / destination_rank["Previous_Searches"] * 100,
        np.nan,
    )
    destination_rank["Trend"] = np.select(
        [
            destination_rank["Search Change %"].gt(0),
            destination_rank["Search Change %"].lt(0),
            destination_rank["Search Change %"].eq(0),
        ],
        ["↑ More", "↓ Less", "→ Same"],
        default="New",
    )
    destination_rank = destination_rank.drop(columns="Previous_Searches")
    st.dataframe(
        style_table(destination_rank[["Destination", "Searches", "Trend", "Search Change %", "Views", "Hotels"]]),
        hide_index=True,
        width="stretch",
        height=354,
        column_config={
            "Searches": st.column_config.NumberColumn(format="localized"),
            "Search Change %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Views": st.column_config.NumberColumn(format="localized"),
        },
    )

st.subheader("Snapshot-week demand change" if view_mode == "Snapshot Week" else "Daily demand change")
daily_display = filtered_daily.rename(
    columns={
        "checkin_date": "Check-in Date",
        "Signal": "Destination Signal",
        "Latest_Searches": "Observed Searches",
        "Change_Pct": "Change %",
        "Hotels_Rising": "Hotels with Rising Views",
        "Action_Hotels": "Action Hotels",
    }
)
daily_display["Change %"] = daily_display["Change %"] * 100
st.dataframe(
    style_table(daily_display[
        [
            "Check-in Date",
            "Destination Signal",
            "Observed Searches",
            "Change %",
            "Hotels with Rising Views",
            "Action Hotels",
        ]
    ]),
    hide_index=True,
    width="stretch",
    height=320,
    column_config={
        "Check-in Date": st.column_config.DateColumn(format="DD MMM YYYY"),
        "Observed Searches": st.column_config.NumberColumn(format="localized"),
        "Change %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Hotels with Rising Views": st.column_config.NumberColumn(format="localized"),
        "Action Hotels": st.column_config.NumberColumn(format="localized"),
    },
)

st.subheader(f"Hotels to check — {selected_ts:%d %b %Y}")
st.caption(
    f"Ranked first by observed hotel views in the comparison period; signals use the {comparison_label}. "
    "The dashboard does not show inventory or parity results."
)

hotel_signal_order = {"Critical surge": 5, "High increase": 4, "New interest": 3, "Stable": 2, "Declining": 1}
destination_signal_order = {"Critical surge": 5, "High increase": 4, "New demand": 3, "Stable": 2, "Declining": 1}
selected["Hotel Signal Order"] = selected["Hotel Signal"].map(hotel_signal_order).fillna(0)
selected["Destination Signal Order"] = selected["Destination Signal"].map(destination_signal_order).fillna(0)
selected["Has Views"] = selected["view_volume"].gt(0)
selected = selected.sort_values(
    ["Has Views", "view_volume", "Hotel Signal Order", "Destination Signal Order", "View_Change"],
    ascending=[False, False, False, False, False],
).reset_index(drop=True)
selected["Priority"] = np.arange(1, len(selected) + 1)
high_views = selected["view_volume"].gt(0) & selected["view_volume"].ge(
    selected["view_volume"].quantile(0.80)
)
rising_views = selected["View_Change"].gt(0)
selected["Work Priority"] = np.select(
    [high_views & rising_views, high_views, rising_views],
    [
        "Urgent — high views and rising",
        "High — high views, even if stable",
        "Medium — lower views but rising",
    ],
    default="Routine — low or zero views without growth",
)

display = selected.rename(
    columns={
        "ProductName": "Hotel",
        "view_volume": "Observed Views",
        "View Change %": "View Change %",
    }
)
display_columns = [
    "Priority",
    "Hotel",
    "Observed Views",
    "View Change %",
    "Hotel Signal",
    "Work Priority",
]
display["View Change %"] = display["View Change %"] * 100
st.dataframe(
    style_table(display[display_columns].head(200)),
    hide_index=True,
    width="stretch",
    height=530,
    column_config={
        "Priority": st.column_config.NumberColumn(format="%d", width="small"),
        "Observed Views": st.column_config.NumberColumn(format="localized"),
        "View Change %": st.column_config.NumberColumn(format="%+.1f%%"),
        "Hotel Signal": st.column_config.TextColumn(width="medium"),
        "Work Priority": st.column_config.TextColumn(width="large"),
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
