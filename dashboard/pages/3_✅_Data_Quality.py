from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils.data import demand_source_path, load_demand, load_engine, load_funnel
from dashboard.utils.ui import apply_theme, style_table
from scripts.pipeline import RAW_DIR, REQUIRED_DEMAND_COLUMNS


st.set_page_config(page_title="HCI | Data Quality", page_icon="✅", layout="wide")
apply_theme()


@st.cache_data(show_spinner=False)
def quality_profile() -> dict[str, object]:
    demand_path = demand_source_path()
    master_path = RAW_DIR / "Master_Hotel.xlsx"

    raw_demand = pd.read_csv(demand_path)
    raw_demand["_product_id"] = pd.to_numeric(raw_demand["ProductID"], errors="coerce")
    raw_demand["_checkin_date"] = pd.to_datetime(raw_demand["CheckInDate"], errors="coerce")
    raw_demand["_snapshot_at"] = pd.to_datetime(
        raw_demand["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False),
        format="mixed",
        errors="coerce",
    )
    master = pd.read_excel(master_path)
    master["_product_id"] = pd.to_numeric(master["ProductID"], errors="coerce")
    master_conflict_columns = [
        column
        for column in ["ProductName", "Destination", "Region", "HotelType Short"]
        if column in master
    ]
    conflicting_master_ids = 0
    duplicated_master = master[
        master["_product_id"].duplicated(False) & master["_product_id"].notna()
    ]
    for _, group in duplicated_master.groupby("_product_id"):
        if any(
            group[column].fillna("<NA>").astype(str).nunique() > 1
            for column in master_conflict_columns
        ):
            conflicting_master_ids += 1

    demand_ids = set(raw_demand["_product_id"].dropna().astype(int))
    master_ids = set(master["_product_id"].dropna().astype(int))
    duplicate_snapshot_excess = int(
        raw_demand.duplicated(["_product_id", "_checkin_date", "_snapshot_at"]).sum()
    )

    return {
        "raw_rows": len(raw_demand),
        "raw_hotels": len(demand_ids),
        "missing_product_ids": int(raw_demand["_product_id"].isna().sum()),
        "invalid_dates": int(raw_demand["_checkin_date"].isna().sum()),
        "invalid_timestamps": int(raw_demand["_snapshot_at"].isna().sum()),
        "duplicate_snapshot_excess": duplicate_snapshot_excess,
        "unmatched_master_hotels": len(demand_ids - master_ids),
        "master_duplicate_excess": int(master.duplicated("ProductID").sum()),
        "conflicting_master_ids": conflicting_master_ids,
        "required_columns_present": REQUIRED_DEMAND_COLUMNS.issubset(raw_demand.columns),
        "latest_snapshot": raw_demand["_snapshot_at"].max(),
        "raw_checkin_min": raw_demand["_checkin_date"].min(),
        "raw_checkin_max": raw_demand["_checkin_date"].max(),
    }


def status_badge(status: str) -> str:
    tone = {"Passed": "success", "Warning": "warning", "Failed": "danger"}.get(status, "neutral")
    return f'<span class="tool-status {tone}">{status}</span>'


def modified_at(path: Path) -> pd.Timestamp | None:
    return pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime)) if path.exists() else None


engine = load_engine()
demand = load_demand()
funnel = load_funnel()
profile = quality_profile()

with st.sidebar:
    st.markdown(
        '<div class="hci-brand"><b>HCI</b><span>HOTEL COMMERCIAL<br>INTELLIGENCE</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Demand-led hotel prioritization")

head_left, head_right = st.columns([4, 1])
with head_left:
    st.title("Data Quality")
    st.caption("Confirm the demand data is fresh and reliable before using hotel priorities.")
with head_right:
    st.markdown(
        '<div class="profile-card"><b>Market Manager</b></div>',
        unsafe_allow_html=True,
    )

funnel_unique = not funnel.duplicated(["ProductID", "checkin_date"]).any()
warning_count = sum(
    [
        profile["raw_rows"] == 750_000,
        profile["missing_product_ids"] > 0,
        profile["duplicate_snapshot_excess"] > 0,
        profile["unmatched_master_hotels"] > 0,
        profile["master_duplicate_excess"] > 0,
        profile["conflicting_master_ids"] > 0,
    ]
)
failed_count = sum(
    [
        not profile["required_columns_present"],
        profile["invalid_dates"] > 0,
        profile["invalid_timestamps"] > 0,
        not funnel_unique,
    ]
)

if failed_count:
    st.markdown(
        f'<div class="quality-banner danger"><b>Data requires attention before use</b> '
        f'· {failed_count} failed checks</div>',
        unsafe_allow_html=True,
    )
elif warning_count:
    st.markdown(
        f'<div class="quality-banner warning"><b>Data is usable with review</b> '
        f'· {warning_count} source warnings · processed hotel/date keys are valid</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="quality-banner success"><b>Data is ready to use</b> · all quality checks passed</div>',
        unsafe_allow_html=True,
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("Demand Records", f"{profile['raw_rows']:,}")
k2.metric("Hotels", f"{profile['raw_hotels']:,}")
k3.metric(
    "Check-in Coverage",
    f"{profile['raw_checkin_min']:%d %b}–{profile['raw_checkin_max']:%d %b}",
)
k4.metric("Latest Snapshot", f"{profile['latest_snapshot']:%d %b, %H:%M}")

checks = pd.DataFrame(
    [
        {
            "Check": "Required demand columns",
            "Status": "Passed" if profile["required_columns_present"] else "Failed",
            "Finding": "All required fields are present"
            if profile["required_columns_present"]
            else "One or more required fields are missing",
            "Action": "None" if profile["required_columns_present"] else "Correct the demand export",
        },
        {
            "Check": "Raw export row limit",
            "Status": "Warning" if profile["raw_rows"] == 750_000 else "Passed",
            "Finding": f"{profile['raw_rows']:,} rows"
            + ("; exact export-limit pattern" if profile["raw_rows"] == 750_000 else ""),
            "Action": "Verify the source export was not truncated"
            if profile["raw_rows"] == 750_000
            else "None",
        },
        {
            "Check": "Missing Product IDs",
            "Status": "Warning" if profile["missing_product_ids"] else "Passed",
            "Finding": f"{profile['missing_product_ids']:,} rows",
            "Action": "Exclude and review affected rows" if profile["missing_product_ids"] else "None",
        },
        {
            "Check": "Duplicate snapshot keys",
            "Status": "Warning" if profile["duplicate_snapshot_excess"] else "Passed",
            "Finding": f"{profile['duplicate_snapshot_excess']:,} excess rows",
            "Action": "Deduplicate during processing" if profile["duplicate_snapshot_excess"] else "None",
        },
        {
            "Check": "Unmatched master hotels",
            "Status": "Warning" if profile["unmatched_master_hotels"] else "Passed",
            "Finding": f"{profile['unmatched_master_hotels']:,} demand hotels",
            "Action": "Update the hotel master" if profile["unmatched_master_hotels"] else "None",
        },
        {
            "Check": "Duplicate master Product IDs",
            "Status": "Warning" if profile["master_duplicate_excess"] else "Passed",
            "Finding": f"{profile['master_duplicate_excess']:,} excess rows",
            "Action": "Define the current master record" if profile["master_duplicate_excess"] else "None",
        },
        {
            "Check": "Conflicting master attributes",
            "Status": "Warning" if profile["conflicting_master_ids"] else "Passed",
            "Finding": f"{profile['conflicting_master_ids']:,} Product IDs have conflicting attributes",
            "Action": "Choose one authoritative master record per Product ID"
            if profile["conflicting_master_ids"]
            else "None",
        },
        {
            "Check": "Valid demand dates",
            "Status": "Failed" if profile["invalid_dates"] or profile["invalid_timestamps"] else "Passed",
            "Finding": f"{profile['invalid_dates']:,} invalid check-in dates; "
            f"{profile['invalid_timestamps']:,} invalid timestamps",
            "Action": "Correct invalid dates" if profile["invalid_dates"] or profile["invalid_timestamps"] else "None",
        },
        {
            "Check": "Hotel/date join validation",
            "Status": "Passed" if funnel_unique else "Failed",
            "Finding": "Processed ProductID + check-in date keys are unique"
            if funnel_unique
            else "Duplicate processed keys found",
            "Action": "None" if funnel_unique else "Stop and repair pipeline joins",
        },
    ]
)

left, right = st.columns([1.75, 1])
with left:
    st.subheader("Quality Checks")
    st.dataframe(
        style_table(checks),
        hide_index=True,
        width="stretch",
        height=390,
        column_config={
            "Check": st.column_config.TextColumn(width="medium"),
            "Status": st.column_config.TextColumn(width="small"),
            "Finding": st.column_config.TextColumn(width="large"),
            "Action": st.column_config.TextColumn(width="large"),
        },
    )

with right:
    st.subheader("Source Freshness")
    source_files = [
        ("Demand", demand_source_path()),
        ("Previous demand", RAW_DIR / "demand_previous.csv.gz"),
        ("Hotel master", RAW_DIR / "Master_Hotel.xlsx"),
        ("Booking production", RAW_DIR / "Booking_Production.csv"),
        ("Internal parity", RAW_DIR / "internal price gap.csv"),
        ("Agoda parity", RAW_DIR / "agoda price gap.xlsx"),
    ]
    freshness_rows = []
    for source, path in source_files:
        stamp = modified_at(path)
        freshness_rows.append(
            {
                "Source": source,
                "File Modified": stamp.strftime("%d %b %Y, %H:%M") if stamp is not None else "Missing",
                "Status": "Available" if stamp is not None else "Missing",
            }
        )
    st.dataframe(style_table(pd.DataFrame(freshness_rows)), hide_index=True, width="stretch", height=250)
    st.caption("File modified time confirms the local file version, not when the upstream source generated it.")

coverage_hotels = funnel.groupby("checkin_date", as_index=False).agg(
    Hotels=("ProductID", "nunique")
)
coverage_searches = (
    funnel.drop_duplicates(["Destination", "checkin_date"])
    .groupby("checkin_date", as_index=False)
    .agg(Destination_Searches=("Destination_Searches", "sum"))
)
coverage = coverage_hotels.merge(coverage_searches, on="checkin_date", how="left").sort_values(
    "checkin_date"
)
max_hotels = coverage["Hotels"].max()
coverage["Coverage"] = pd.cut(
    coverage["Hotels"],
    bins=[-1, max_hotels * 0.30, max_hotels * 0.75, float("inf")],
    labels=["Low", "Partial", "Broad"],
)

coverage_col, attention_col = st.columns([1.75, 1])
with coverage_col:
    st.subheader("Coverage by Check-in Date")
    coverage_chart = (
        alt.Chart(coverage)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("checkin_date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=-35)),
            y=alt.Y("Hotels:Q", title="Hotels represented"),
            color=alt.Color(
                "Coverage:N",
                scale=alt.Scale(
                    domain=["Broad", "Partial", "Low"],
                    range=["#22c55e", "#f59e0b", "#94a3b8"],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("checkin_date:T", title="Check-in", format="%d %b %Y"),
                alt.Tooltip("Hotels:Q", format=",.0f"),
                alt.Tooltip("Destination_Searches:Q", title="Destination searches", format=",.0f"),
                "Coverage:N",
            ],
        )
        .properties(height=290)
    )
    st.altair_chart(coverage_chart, width="stretch")
    st.caption("Coverage describes how many hotels appear for each check-in date; it is not an occupancy measure.")

with attention_col:
    st.subheader("Attention Needed")
    attention = checks[checks["Status"].isin(["Warning", "Failed"])][
        ["Check", "Status", "Action"]
    ]
    if attention.empty:
        st.success("No data-quality actions are currently required.")
    else:
        for item in attention.itertuples(index=False):
            st.markdown(
                f'<div class="date-row"><div><b>{item.Check}</b><br><span>{item.Action}</span></div>'
                f'{status_badge(item.Status)}</div>',
                unsafe_allow_html=True,
            )
        st.download_button(
            "Download issue report",
            attention.to_csv(index=False).encode("utf-8-sig"),
            file_name="hci_data_quality_issues.csv",
            mime="text/csv",
        )

st.markdown(
    '<div class="workflow-note"><b>How to use this page:</b> Warnings do not automatically make demand unusable. '
    'Review the affected records and stop only when a check is marked Failed.</div>',
    unsafe_allow_html=True,
)
