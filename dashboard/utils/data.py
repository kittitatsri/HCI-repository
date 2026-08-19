from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def demand_source_path() -> Path:
    for filename in ("demand_latest.csv.gz", "demand_latest.csv"):
        path = ROOT / "data" / "raw" / filename
        if path.exists():
            return path
    raise FileNotFoundError("Missing demand_latest.csv.gz or demand_latest.csv")


def load_engine() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "engine.csv"
    return pd.read_csv(path)


def load_demand() -> pd.DataFrame:
    path = demand_source_path()
    frame = pd.read_csv(path)
    frame["snapshot_at"] = pd.to_datetime(
        frame["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False),
        format="mixed",
        errors="coerce",
    )
    frame["checkin_date"] = pd.to_datetime(frame["CheckInDate"], errors="coerce")
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    return frame


@st.cache_data(show_spinner=False)
def load_demand_history() -> pd.DataFrame:
    """Combine the previous and latest stored demand files for complete trend charts."""
    frames: list[pd.DataFrame] = []
    for stem in ("demand_previous", "demand_latest"):
        path = next(
            (
                ROOT / "data" / "raw" / filename
                for filename in (f"{stem}.csv.gz", f"{stem}.csv")
                if (ROOT / "data" / "raw" / filename).exists()
            ),
            None,
        )
        if path is not None:
            frames.append(pd.read_csv(path))
    if not frames:
        return load_demand()
    frame = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    frame["snapshot_at"] = pd.to_datetime(
        frame["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False),
        format="mixed",
        errors="coerce",
    )
    frame["checkin_date"] = pd.to_datetime(frame["CheckInDate"], errors="coerce")
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    return frame


def iso_week_options(demand: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """Return available ISO snapshot weeks, newest first."""
    valid = demand["snapshot_at"].dropna().dt.normalize()
    starts = valid - pd.to_timedelta(valid.dt.weekday, unit="D")
    unique_starts = sorted(starts.drop_duplicates(), reverse=True)
    options: list[tuple[str, pd.Timestamp]] = []
    for start in unique_starts:
        iso = start.isocalendar()
        end = start + pd.Timedelta(days=6)
        label = f"{iso.year}-W{iso.week:02d} ({start:%d %b}–{end:%d %b})"
        options.append((label, pd.Timestamp(start)))
    return options


@st.cache_data(show_spinner=False)
def build_checkin_date_trend(demand: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all stored interval searches and views by destination/check-in date."""
    frame = demand.copy()
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce")
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce").dt.normalize()
    frame["search_volume"] = pd.to_numeric(frame["search_volume"], errors="coerce")
    frame["view_volume"] = pd.to_numeric(frame["view_volume"], errors="coerce").fillna(0)
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["snapshot_at", "checkin_date", "Destination"])
    timestamps = frame["snapshot_at"].drop_duplicates().sort_values()
    batches = timestamps.diff().dt.total_seconds().fillna(301).gt(300).cumsum()
    batch_map = pd.DataFrame({"snapshot_at": timestamps, "Observation_Batch": batches})
    frame = frame.merge(batch_map, on="snapshot_at", how="left", validate="many_to_one")
    search_interval = (
        frame.groupby(
            ["Observation_Batch", "Destination", "checkin_date"], as_index=False
        )["search_volume"]
        .median()
    )
    searches = (
        search_interval.groupby(["Destination", "checkin_date"], as_index=False)["search_volume"]
        .sum()
        .rename(columns={"search_volume": "Searches"})
    )
    view_interval = (
        frame.dropna(subset=["ProductID"])
        .sort_values(["Observation_Batch", "ProductID", "checkin_date", "snapshot_at"])
        .drop_duplicates(["Observation_Batch", "ProductID", "checkin_date"], keep="last")
    )
    views = (
        view_interval.groupby(["Destination", "checkin_date"], as_index=False)["view_volume"]
        .sum()
        .rename(columns={"view_volume": "Views"})
    )
    return searches.merge(views, on=["Destination", "checkin_date"], how="outer")


@st.cache_data(show_spinner=False)
def build_full_history_funnel(demand: pd.DataFrame) -> pd.DataFrame:
    """Build hotel/check-in demand from every stored observation interval."""
    frame = demand.copy()
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce")
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce").dt.normalize()
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    frame["search_volume"] = pd.to_numeric(frame["search_volume"], errors="coerce")
    frame["view_volume"] = pd.to_numeric(frame["view_volume"], errors="coerce").fillna(0)
    frame["hotness_score"] = pd.to_numeric(frame["hotness_score"], errors="coerce")
    frame["trend_momentum"] = pd.to_numeric(frame["trend_momentum"], errors="coerce")
    frame["check_status"] = (
        frame["check_status"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    frame = frame.dropna(
        subset=["snapshot_at", "checkin_date", "ProductID", "Destination"]
    )
    timestamps = frame["snapshot_at"].drop_duplicates().sort_values()
    batches = timestamps.diff().dt.total_seconds().fillna(301).gt(300).cumsum()
    batch_map = pd.DataFrame({"snapshot_at": timestamps, "Observation_Batch": batches})
    frame = frame.merge(batch_map, on="snapshot_at", how="left", validate="many_to_one")

    hotel_keys = ["ProductID", "checkin_date"]
    hotel_intervals = (
        frame.sort_values(["Observation_Batch"] + hotel_keys + ["snapshot_at"])
        .drop_duplicates(["Observation_Batch"] + hotel_keys, keep="last")
    )
    metadata = (
        hotel_intervals.sort_values(hotel_keys + ["snapshot_at"])
        .drop_duplicates(hotel_keys, keep="last")
        [hotel_keys + ["ProductName", "Destination", "snapshot_at", "check_status"]]
    )
    hotels = (
        hotel_intervals.groupby(hotel_keys, as_index=False)
        .agg(
            view_volume=("view_volume", "sum"),
            hotness_score=("hotness_score", "mean"),
            trend_momentum=("trend_momentum", "mean"),
            Interval_Count=("Observation_Batch", "nunique"),
        )
        .merge(metadata, on=hotel_keys, validate="one_to_one")
    )

    destination_keys = ["Destination", "checkin_date"]
    destination_intervals = (
        frame.groupby(["Observation_Batch"] + destination_keys, as_index=False)
        .agg(search_volume=("search_volume", "median"), snapshot_at=("snapshot_at", "max"))
    )
    destinations = (
        destination_intervals.groupby(destination_keys, as_index=False)
        .agg(
            Destination_Searches=("search_volume", "sum"),
            Destination_Search_Snapshot=("snapshot_at", "max"),
            Destination_Intervals=("Observation_Batch", "nunique"),
        )
    )
    return hotels.merge(destinations, on=destination_keys, how="left", validate="many_to_one")


@st.cache_data(show_spinner=False)
def build_hotel_checkin_trend(demand: pd.DataFrame, product_id: int) -> pd.DataFrame:
    """Aggregate hotel views and its destination-search context by check-in date."""
    frame = demand.copy()
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce")
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce").dt.normalize()
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    frame["view_volume"] = pd.to_numeric(frame["view_volume"], errors="coerce").fillna(0)
    frame["search_volume"] = pd.to_numeric(frame["search_volume"], errors="coerce").fillna(0)
    hotel_frame = frame[
        frame["ProductID"].eq(product_id)
        & frame["snapshot_at"].notna()
        & frame["checkin_date"].notna()
    ].copy()
    if hotel_frame.empty:
        return pd.DataFrame(columns=["checkin_date", "Destination Searches", "Hotel Views"])

    destination = hotel_frame["Destination"].dropna().astype(str).iloc[-1]
    destination_frame = frame[
        frame["Destination"].astype(str).eq(destination)
        & frame["snapshot_at"].notna()
        & frame["checkin_date"].notna()
    ].copy()
    timestamps = frame["snapshot_at"].drop_duplicates().sort_values()
    batches = timestamps.diff().dt.total_seconds().fillna(301).gt(300).cumsum()
    batch_map = pd.DataFrame({"snapshot_at": timestamps, "Observation_Batch": batches})
    hotel_frame = hotel_frame.merge(batch_map, on="snapshot_at", how="left", validate="many_to_one")
    destination_frame = destination_frame.merge(
        batch_map, on="snapshot_at", how="left", validate="many_to_one"
    )
    interval = (
        hotel_frame.sort_values(["Observation_Batch", "checkin_date", "snapshot_at"])
        .drop_duplicates(["Observation_Batch", "checkin_date"], keep="last")
    )
    views = (
        interval.groupby("checkin_date", as_index=False)["view_volume"]
        .sum()
        .rename(columns={"view_volume": "Hotel Views"})
        .sort_values("checkin_date")
    )
    searches = (
        destination_frame.groupby(
            ["Observation_Batch", "checkin_date"], as_index=False
        )["search_volume"]
        .median()
        .groupby("checkin_date", as_index=False)["search_volume"]
        .sum()
        .rename(columns={"search_volume": "Destination Searches"})
    )
    return searches.merge(views, on="checkin_date", how="outer").sort_values("checkin_date")


@st.cache_data(show_spinner=False)
def build_weekly_comparison(demand: pd.DataFrame, week_start: pd.Timestamp) -> pd.DataFrame:
    """Compare summed demand intervals in one ISO snapshot week with the prior week."""
    frame = demand.copy()
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce")
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce").dt.normalize()
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    frame["search_volume"] = pd.to_numeric(frame["search_volume"], errors="coerce")
    frame["view_volume"] = pd.to_numeric(frame["view_volume"], errors="coerce").fillna(0)
    frame["check_status"] = (
        frame["check_status"].fillna("unknown").astype(str).str.strip().str.lower()
    )

    selected_start = pd.Timestamp(week_start).normalize()
    selected_end = selected_start + pd.Timedelta(days=7)
    previous_start = selected_start - pd.Timedelta(days=7)
    relevant = frame[
        frame["snapshot_at"].ge(previous_start)
        & frame["snapshot_at"].lt(selected_end)
        & frame["checkin_date"].notna()
    ].copy()
    relevant["period"] = np.where(
        relevant["snapshot_at"].ge(selected_start), "latest", "previous"
    )

    timestamps = relevant["snapshot_at"].drop_duplicates().sort_values()
    batches = timestamps.diff().dt.total_seconds().fillna(301).gt(300).cumsum()
    batch_map = pd.DataFrame({"snapshot_at": timestamps, "Observation_Batch": batches})
    relevant = relevant.merge(batch_map, on="snapshot_at", how="left", validate="many_to_one")
    interval_counts = relevant.groupby("period")["Observation_Batch"].nunique()
    latest_count = int(interval_counts.get("latest", 0))
    previous_count = int(interval_counts.get("previous", 0))
    previous_scale = latest_count / previous_count if previous_count else np.nan

    hotel_keys = ["ProductID", "checkin_date"]
    hotel_rows = (
        relevant.dropna(subset=["ProductID"])
        .sort_values(["period", "Observation_Batch"] + hotel_keys + ["snapshot_at"])
        .drop_duplicates(["period", "Observation_Batch"] + hotel_keys, keep="last")
    )
    metadata = (
        hotel_rows[hotel_rows["period"].eq("latest")]
        .sort_values(hotel_keys + ["snapshot_at"])
        .drop_duplicates(hotel_keys, keep="last")
        [hotel_keys + ["ProductName", "Destination", "snapshot_at", "check_status"]]
    )
    latest_hotels = (
        hotel_rows[hotel_rows["period"].eq("latest")]
        .groupby(hotel_keys, as_index=False)
        .agg(
            view_volume=("view_volume", "sum"),
            hotness_score=("hotness_score", "mean"),
            trend_momentum=("trend_momentum", "mean"),
        )
        .merge(metadata, on=hotel_keys, validate="one_to_one")
    )
    previous_hotels = (
        hotel_rows[hotel_rows["period"].eq("previous")]
        .groupby(hotel_keys, as_index=False)["view_volume"]
        .sum()
        .rename(columns={"view_volume": "Previous_Views"})
    )
    if pd.notna(previous_scale):
        previous_hotels["Previous_Views"] *= previous_scale
    latest_hotels = latest_hotels.merge(
        previous_hotels, on=hotel_keys, how="left", validate="one_to_one"
    )
    latest_hotels["View_Change"] = latest_hotels["view_volume"] - latest_hotels["Previous_Views"]

    destination_keys = ["Destination", "checkin_date"]
    destination_intervals = (
        relevant.dropna(subset=["Destination"])
        .groupby(["period", "Observation_Batch"] + destination_keys, as_index=False)
        .agg(search_volume=("search_volume", "median"), snapshot_at=("snapshot_at", "max"))
    )
    latest_destinations = (
        destination_intervals[destination_intervals["period"].eq("latest")]
        .groupby(destination_keys, as_index=False)
        .agg(
            Destination_Searches=("search_volume", "sum"),
            Destination_Search_Snapshot=("snapshot_at", "max"),
        )
    )
    previous_destinations = (
        destination_intervals[destination_intervals["period"].eq("previous")]
        .groupby(destination_keys, as_index=False)["search_volume"]
        .sum()
        .rename(columns={"search_volume": "Previous_Destination_Searches"})
    )
    if pd.notna(previous_scale):
        previous_destinations["Previous_Destination_Searches"] *= previous_scale
    destinations = latest_destinations.merge(
        previous_destinations, on=destination_keys, how="left", validate="one_to_one"
    )
    destinations["Previous_Destination_Search_Snapshot"] = selected_start - pd.Timedelta(seconds=1)
    destinations["Destination_Search_Change"] = (
        destinations["Destination_Searches"] - destinations["Previous_Destination_Searches"]
    )

    result = latest_hotels.merge(
        destinations, on=destination_keys, how="left", validate="many_to_one"
    )
    result["Upload_Change_Status"] = np.select(
        [result["Previous_Views"].isna(), result["View_Change"].gt(0), result["View_Change"].lt(0)],
        ["New", "Increase", "Decrease"],
        default="No change",
    )
    return result


@st.cache_data(show_spinner=False)
def build_checkin_week_comparison(
    detail: pd.DataFrame, week_start: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare one ISO check-in week with the immediately preceding check-in week."""
    frame = detail.copy()
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce").dt.normalize()
    frame["view_volume"] = pd.to_numeric(frame["view_volume"], errors="coerce").fillna(0)
    frame["Destination_Searches"] = pd.to_numeric(
        frame["Destination_Searches"], errors="coerce"
    ).fillna(0)
    selected_start = pd.Timestamp(week_start).normalize()
    selected_end = selected_start + pd.Timedelta(days=7)
    previous_start = selected_start - pd.Timedelta(days=7)

    current = frame[
        frame["checkin_date"].ge(selected_start) & frame["checkin_date"].lt(selected_end)
    ].copy()
    previous = frame[
        frame["checkin_date"].ge(previous_start) & frame["checkin_date"].lt(selected_start)
    ].copy()

    def destination_week(source: pd.DataFrame, value_name: str) -> pd.DataFrame:
        return (
            source.drop_duplicates(["Destination", "checkin_date"])
            .groupby("Destination", as_index=False)["Destination_Searches"]
            .sum()
            .rename(columns={"Destination_Searches": value_name})
        )

    destinations = destination_week(current, "Current_Searches").merge(
        destination_week(previous, "Previous_Searches"),
        on="Destination",
        how="outer",
        validate="one_to_one",
    )
    destinations[["Current_Searches", "Previous_Searches"]] = destinations[
        ["Current_Searches", "Previous_Searches"]
    ].fillna(0)
    destinations["Search_Change"] = (
        destinations["Current_Searches"] - destinations["Previous_Searches"]
    )
    destinations["Change_Pct"] = np.where(
        destinations["Previous_Searches"].gt(0),
        destinations["Search_Change"] / destinations["Previous_Searches"],
        np.nan,
    )

    current_hotels = current.groupby("ProductID", as_index=False).agg(
        ProductName=("ProductName", "last"),
        Destination=("Destination", "last"),
        Current_Views=("view_volume", "sum"),
        CheckIn_Days=("checkin_date", "nunique"),
    )
    previous_hotels = previous.groupby("ProductID", as_index=False).agg(
        Previous_Views=("view_volume", "sum")
    )
    hotels = current_hotels.merge(
        previous_hotels, on="ProductID", how="left", validate="one_to_one"
    )
    hotels["View_Change"] = hotels["Current_Views"] - hotels["Previous_Views"]
    hotels["View_Change_Pct"] = np.where(
        hotels["Previous_Views"].gt(0),
        hotels["View_Change"] / hotels["Previous_Views"],
        np.nan,
    )
    hotels = hotels.merge(
        destinations[
            ["Destination", "Current_Searches", "Previous_Searches", "Search_Change", "Change_Pct"]
        ],
        on="Destination",
        how="left",
        validate="many_to_one",
    )

    def daily_searches(source: pd.DataFrame, period: str, shift_days: int = 0) -> pd.DataFrame:
        result = (
            source.drop_duplicates(["Destination", "checkin_date"])
            .groupby("checkin_date", as_index=False)["Destination_Searches"]
            .sum()
            .rename(columns={"Destination_Searches": "Searches"})
        )
        result["Matched_Date"] = result["checkin_date"] + pd.Timedelta(days=shift_days)
        result["Period"] = period
        return result

    daily = pd.concat(
        [
            daily_searches(current, "Selected week"),
            daily_searches(previous, "Previous week", shift_days=7),
        ],
        ignore_index=True,
    )
    daily["Weekday"] = pd.Categorical(
        daily["Matched_Date"].dt.day_name().str[:3],
        categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ordered=True,
    )
    return hotels, destinations, daily


def load_funnel() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "hotel_date_funnel.csv"
    frame = pd.read_csv(path)
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce")
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce")
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    return frame


def clear_data_cache() -> None:
    st.cache_data.clear()
