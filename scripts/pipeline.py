from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"

REQUIRED_DEMAND_COLUMNS = {
    "Time Stamp",
    "CheckInDate",
    "ProductID",
    "ProductName",
    "Destination",
    "trend_momentum",
    "hotness_score",
    "check_status",
    "search_volume",
    "view_volume",
}

BOOKING_FILE = RAW_DIR / "Booking_Production.csv"
INTERNAL_PARITY_FILE = RAW_DIR / "internal price gap.csv"
AGODA_PARITY_FILE = RAW_DIR / "agoda price gap.xlsx"
DEMAND_FILENAMES = ("demand_latest.csv.gz", "demand_latest.csv")
PREVIOUS_DEMAND_FILENAMES = ("demand_previous.csv.gz", "demand_previous.csv")
DEMAND_EVENT_KEYS = ["Time Stamp", "CheckInDate", "ProductID"]
VALID_CHECK_STATUSES = {"continues", "new entry", "modify"}


def _parse_demand_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str.replace("\u202f", " ", regex=False),
        format="mixed",
        errors="coerce",
    )


def resolve_demand_path() -> Path:
    for filename in DEMAND_FILENAMES:
        path = RAW_DIR / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing demand source. Add {DEMAND_FILENAMES[0]} or {DEMAND_FILENAMES[1]} to {RAW_DIR}"
    )


def resolve_previous_demand_path() -> Path | None:
    for filename in PREVIOUS_DEMAND_FILENAMES:
        path = RAW_DIR / filename
        if path.exists():
            return path
    return None


def merge_incremental_demand(
    history: pd.DataFrame, incremental: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int | bool | pd.Timestamp]]:
    """Append one incremental export to demand history without double-counting events."""
    for label, frame in (("Current history", history), ("Uploaded file", incremental)):
        missing = REQUIRED_DEMAND_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")

    incoming = incremental.copy()
    missing_product = pd.to_numeric(incoming["ProductID"], errors="coerce").isna()
    skipped_missing_product = int(missing_product.sum())
    incoming = incoming.loc[~missing_product].copy()
    if incoming.empty:
        raise ValueError("Every uploaded row is missing ProductID; there is no demand data to merge.")
    incoming_status = incoming["check_status"].fillna("").astype(str).str.strip().str.lower()
    invalid_statuses = sorted(
        set(incoming_status[incoming_status.ne("")]).difference(VALID_CHECK_STATUSES)
    )
    if invalid_statuses:
        raise ValueError(
            "Uploaded check_status contains unsupported values: " + ", ".join(invalid_statuses)
        )

    combined = pd.concat([history, incoming], ignore_index=True, sort=False)
    combined["_snapshot_key"] = _parse_demand_timestamp(combined["Time Stamp"])
    combined["_checkin_key"] = pd.to_datetime(combined["CheckInDate"], errors="coerce").dt.normalize()
    combined["_product_key"] = pd.to_numeric(combined["ProductID"], errors="coerce").astype("Int64")

    invalid_key = combined[["_snapshot_key", "_checkin_key", "_product_key"]].isna().any(axis=1)
    history_invalid = int(invalid_key.iloc[: len(history)].sum())
    upload_invalid = int(invalid_key.iloc[len(history) :].sum())
    if upload_invalid:
        raise ValueError(
            f"The uploaded file contains {upload_invalid:,} rows with an invalid Time Stamp, "
            "CheckInDate, or ProductID. Correct these rows before uploading."
        )
    # Preserve legacy invalid history rows without allowing their missing keys to collapse together.
    combined["_row_fallback"] = 0
    if history_invalid:
        combined.loc[invalid_key, "_row_fallback"] = combined.index[invalid_key] + 1

    history_clean = (
        combined.iloc[: len(history)]
        .drop_duplicates(
            ["_snapshot_key", "_checkin_key", "_product_key", "_row_fallback"], keep="last"
        )
        .sort_values("_snapshot_key", kind="stable")
    )
    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        ["_snapshot_key", "_checkin_key", "_product_key", "_row_fallback"], keep="last"
    ).sort_values("_snapshot_key", kind="stable")
    duplicate_rows = before_dedup - len(combined)
    history_keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "snapshot": _parse_demand_timestamp(history["Time Stamp"]),
                "checkin": pd.to_datetime(history["CheckInDate"], errors="coerce").dt.normalize(),
                "product": pd.to_numeric(history["ProductID"], errors="coerce").astype("Int64"),
            }
        )
    )
    incoming_keys = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "snapshot": _parse_demand_timestamp(incoming["Time Stamp"]),
                "checkin": pd.to_datetime(incoming["CheckInDate"], errors="coerce").dt.normalize(),
                "product": pd.to_numeric(incoming["ProductID"], errors="coerce").astype("Int64"),
            }
        )
    )
    new_events = int((~incoming_keys.unique().isin(history_keys)).sum())

    result = combined.drop(
        columns=["_snapshot_key", "_checkin_key", "_product_key", "_row_fallback"]
    )
    result = result.reindex(columns=list(history.columns) + [c for c in result.columns if c not in history.columns])
    comparable_history = history_clean.drop(
        columns=["_snapshot_key", "_checkin_key", "_product_key", "_row_fallback"]
    ).reindex(columns=result.columns)
    data_changed = not result.reset_index(drop=True).equals(comparable_history.reset_index(drop=True))
    stats: dict[str, int | bool | pd.Timestamp] = {
        "history_rows": len(history),
        "uploaded_rows": len(incremental),
        "skipped_missing_product": skipped_missing_product,
        "new_events": new_events,
        "duplicates_removed": duplicate_rows,
        "merged_rows": len(result),
        "data_changed": data_changed,
        "upload_min_timestamp": _parse_demand_timestamp(incoming["Time Stamp"]).min(),
        "upload_max_timestamp": _parse_demand_timestamp(incoming["Time Stamp"]).max(),
    }
    return result, stats


def _safe_max_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    maximum = values.max()
    return values * 0 if not maximum else (values / maximum * 100)


def _normalise_status(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unknown", "null": "unknown", "": "unknown"})
        .str.title()
    )


def load_sources(demand_path: Path | None = None):
    if demand_path is None:
        try:
            demand_path = resolve_demand_path()
        except FileNotFoundError:
            candidates = []
            for path in [*RAW_DIR.glob("*.csv"), *RAW_DIR.glob("*.csv.gz")]:
                columns = set(pd.read_csv(path, nrows=0).columns)
                if REQUIRED_DEMAND_COLUMNS.issubset(columns):
                    candidates.append(path)
            if not candidates:
                raise FileNotFoundError(f"No valid demand CSV found in {RAW_DIR}")
            demand_path = max(candidates, key=lambda path: path.stat().st_mtime)

    demand = pd.read_csv(demand_path)
    missing = REQUIRED_DEMAND_COLUMNS.difference(demand.columns)
    if missing:
        raise ValueError(f"Demand CSV is missing required columns: {', '.join(sorted(missing))}")

    master = pd.read_excel(RAW_DIR / "Master_Hotel.xlsx")
    performance = pd.read_excel(RAW_DIR / "Hotel_Performance.xlsx")
    return demand, master, performance, Path(demand_path)


def prepare_demand(demand: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy()
    demand["ProductID"] = pd.to_numeric(demand["ProductID"], errors="coerce").astype("Int64")
    demand["snapshot_at"] = pd.to_datetime(
        demand["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False),
        format="mixed",
        errors="coerce",
    )
    demand["checkin_date"] = pd.to_datetime(demand["CheckInDate"], errors="coerce")
    for column in ["trend_momentum", "hotness_score", "search_volume", "view_volume"]:
        demand[column] = pd.to_numeric(demand[column], errors="coerce")
    demand["check_status"] = demand["check_status"].fillna("unknown").astype(str).str.strip().str.lower()
    return demand.dropna(subset=["ProductID", "ProductName"])


def _with_observation_batches(demand: pd.DataFrame) -> pd.DataFrame:
    """Group near-identical crawler timestamps into one demand observation interval."""
    frame = demand.drop(
        columns=["Observation_Batch", "Observation_At"], errors="ignore"
    ).dropna(subset=["snapshot_at"]).copy()
    timestamps = frame["snapshot_at"].drop_duplicates().sort_values()
    batches = timestamps.diff().dt.total_seconds().fillna(301).gt(300).cumsum()
    batch_map = pd.DataFrame({"snapshot_at": timestamps, "Observation_Batch": batches})
    frame = frame.merge(batch_map, on="snapshot_at", how="left", validate="many_to_one")
    batch_times = frame.groupby("Observation_Batch")["snapshot_at"].transform("max")
    frame["Observation_At"] = batch_times
    return frame


def select_interval_periods(
    demand: pd.DataFrame, previous_demand: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Select the newest uploaded intervals and an equally sized preceding baseline."""
    current_all = _with_observation_batches(demand)
    previous_all = (
        _with_observation_batches(previous_demand)
        if previous_demand is not None and not previous_demand.empty
        else None
    )

    if previous_all is not None:
        previous_timestamps = set(previous_all["snapshot_at"].dropna())
        new_rows = current_all.loc[~current_all["snapshot_at"].isin(previous_timestamps)]
    else:
        new_rows = current_all.iloc[0:0]

    if not new_rows.empty:
        current_start = new_rows["snapshot_at"].min()
        current_period = current_all[current_all["snapshot_at"].ge(current_start)].copy()
        baseline_source = previous_all
    else:
        latest_day = current_all["snapshot_at"].max().normalize()
        current_period = current_all[current_all["snapshot_at"].dt.normalize().eq(latest_day)].copy()
        baseline_source = current_all[current_all["snapshot_at"].lt(latest_day)].copy()

    interval_count = current_period["Observation_Batch"].nunique()
    if baseline_source is None or baseline_source.empty or interval_count == 0:
        return current_period, None
    baseline_batches = (
        baseline_source[["Observation_Batch", "Observation_At"]]
        .drop_duplicates()
        .sort_values("Observation_At")
        .tail(interval_count)["Observation_Batch"]
    )
    baseline_period = baseline_source[
        baseline_source["Observation_Batch"].isin(baseline_batches)
    ].copy()
    return current_period, baseline_period


def build_latest_demand_by_checkin(demand: pd.DataFrame) -> pd.DataFrame:
    """Sum interval views into one hotel/check-in record for the selected observation period."""
    keys = ["ProductID", "checkin_date"]
    interval = _with_observation_batches(demand)
    interval = (
        interval.dropna(subset=keys)
        .sort_values(["Observation_Batch"] + keys + ["snapshot_at"])
        .drop_duplicates(["Observation_Batch"] + keys, keep="last")
    )
    latest_metadata = (
        interval.sort_values(keys + ["snapshot_at"])
        .drop_duplicates(keys, keep="last")
        [keys[0:1] + ["checkin_date", "ProductName", "Destination", "check_status"]]
    )
    aggregated = interval.groupby(keys, as_index=False).agg(
        snapshot_at=("snapshot_at", "max"),
        hotness_score=("hotness_score", "mean"),
        trend_momentum=("trend_momentum", "mean"),
        search_volume=("search_volume", "median"),
        view_volume=("view_volume", "sum"),
        Interval_Count=("Observation_Batch", "nunique"),
    )
    return aggregated.merge(latest_metadata, on=keys, validate="one_to_one")


def build_latest_destination_demand(demand: pd.DataFrame) -> pd.DataFrame:
    """Sum one robust destination-search observation per interval and check-in date."""
    keys = ["Destination", "checkin_date"]
    interval = _with_observation_batches(demand).dropna(subset=keys)
    interval_search = (
        interval.groupby(["Observation_Batch", "Observation_At"] + keys, as_index=False)
        .agg(search_volume=("search_volume", "median"))
    )
    destination = interval_search.groupby(keys, as_index=False).agg(
        Destination_Search_Snapshot=("Observation_At", "max"),
        Destination_Searches=("search_volume", "sum"),
        Destination_Intervals=("Observation_Batch", "nunique"),
    )
    return destination.rename(
        columns={
            "Destination_Searches": "Destination_Searches",
        }
    )


def load_booking_production() -> pd.DataFrame:
    if not BOOKING_FILE.exists():
        return pd.DataFrame(columns=["ProductID", "checkin_date", "Bookings"])
    booking = pd.read_csv(BOOKING_FILE)
    booking["ProductID"] = pd.to_numeric(booking["ProductID"], errors="coerce").astype("Int64")
    booking["checkin_date"] = pd.to_datetime(
        booking["CheckInDate"], format="mixed", errors="coerce"
    ).dt.normalize()
    booking["booking_count"] = pd.to_numeric(booking["booking_count"], errors="coerce").fillna(0)
    return (
        booking.dropna(subset=["ProductID", "checkin_date"])
        .groupby(["ProductID", "checkin_date"], as_index=False)
        .agg(Bookings=("booking_count", "sum"))
    )


def load_internal_parity() -> pd.DataFrame:
    if not INTERNAL_PARITY_FILE.exists():
        return pd.DataFrame(
            columns=[
                "ProductID",
                "stay_date",
                "Internal_Worst_Gap",
                "Internal_Median_Gap",
                "Internal_Comparable_Rates",
                "Internal_Total_Rates",
            ]
        )
    parity = pd.read_csv(INTERNAL_PARITY_FILE)
    parity["ProductID"] = pd.to_numeric(parity["ProductID"], errors="coerce").astype("Int64")
    parity["stay_date"] = pd.to_datetime(parity["stayDate"], errors="coerce").dt.normalize()
    raw_gap = parity["Market Margin Gap"].astype(str).str.strip().replace({"-": np.nan, "": np.nan})
    parity["gap"] = pd.to_numeric(raw_gap, errors="coerce")
    grouped = parity.dropna(subset=["ProductID", "stay_date"]).groupby(
        ["ProductID", "stay_date"], as_index=False
    )
    result = grouped.agg(
        Internal_Worst_Gap=("gap", "min"),
        Internal_Median_Gap=("gap", "median"),
        Internal_Comparable_Rates=("gap", "count"),
        Internal_Total_Rates=("gap", "size"),
    )
    return result


def load_agoda_parity() -> pd.DataFrame:
    if not AGODA_PARITY_FILE.exists():
        return pd.DataFrame(
            columns=["ProductID", "stay_date", "Agoda_Raw_Gap", "Agoda_Price_Disadvantage"]
        )
    wide = pd.read_excel(AGODA_PARITY_FILE, header=1)
    wide = wide.rename(columns={"Hotel Id": "ProductID"})
    wide["ProductID"] = pd.to_numeric(wide["ProductID"], errors="coerce").astype("Int64")
    id_columns = ["ProductID"]
    date_columns = [column for column in wide.columns if isinstance(column, (pd.Timestamp, __import__("datetime").datetime))]
    long = wide[id_columns + date_columns].melt(
        id_vars=id_columns, var_name="stay_date", value_name="Agoda_Raw_Gap"
    )
    long["stay_date"] = pd.to_datetime(long["stay_date"], errors="coerce").dt.normalize()
    long["Agoda_Raw_Gap"] = pd.to_numeric(long["Agoda_Raw_Gap"], errors="coerce")
    long = (
        long.dropna(subset=["ProductID", "stay_date"])
        .groupby(["ProductID", "stay_date"], as_index=False)["Agoda_Raw_Gap"]
        .min()
    )
    # Agoda export uses 0 for cheapest and negative values for disadvantage.
    long["Agoda_Price_Disadvantage"] = (-long["Agoda_Raw_Gap"]).clip(lower=0)
    return long


def build_hotel_date_funnel(
    demand: pd.DataFrame, previous_demand: pd.DataFrame | None = None
) -> pd.DataFrame:
    current_interval_count = _with_observation_batches(demand)["Observation_Batch"].nunique()
    latest = build_latest_demand_by_checkin(demand)
    latest["checkin_date"] = latest["checkin_date"].dt.normalize()
    latest = latest.rename(columns={"search_volume": "Raw_Destination_Search"})
    latest["view_volume"] = pd.to_numeric(latest["view_volume"], errors="coerce").fillna(0)
    destination_latest = build_latest_destination_demand(demand)
    destination_latest["checkin_date"] = destination_latest["checkin_date"].dt.normalize()
    destination_latest["Destination_Searches"] = pd.to_numeric(
        destination_latest["Destination_Searches"], errors="coerce"
    ).fillna(0)
    funnel = latest.merge(
        destination_latest,
        on=["Destination", "checkin_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        load_booking_production(),
        on=["ProductID", "checkin_date"],
        how="left",
        validate="one_to_one",
    )
    funnel["Bookings"] = funnel["Bookings"].fillna(0)
    funnel = funnel.merge(
        load_internal_parity().rename(columns={"stay_date": "checkin_date"}),
        on=["ProductID", "checkin_date"],
        how="left",
        validate="one_to_one",
    )
    funnel = funnel.merge(
        load_agoda_parity().rename(columns={"stay_date": "checkin_date"}),
        on=["ProductID", "checkin_date"],
        how="left",
        validate="one_to_one",
    )
    funnel["Views_per_Search"] = np.where(
        funnel["Destination_Searches"].gt(0),
        funnel["view_volume"] / funnel["Destination_Searches"],
        np.nan,
    )
    funnel["Search_to_Booking"] = np.where(
        funnel["view_volume"].gt(0),
        funnel["Bookings"] / funnel["view_volume"],
        np.nan,
    )
    funnel["Internal_Comparable_Coverage"] = np.where(
        funnel["Internal_Total_Rates"].gt(0),
        funnel["Internal_Comparable_Rates"] / funnel["Internal_Total_Rates"],
        np.nan,
    )
    if previous_demand is not None and not previous_demand.empty:
        previous_interval_count = _with_observation_batches(previous_demand)[
            "Observation_Batch"
        ].nunique()
        coverage_scale = (
            current_interval_count / previous_interval_count if previous_interval_count else np.nan
        )
        previous = build_latest_demand_by_checkin(previous_demand)[
            ["ProductID", "checkin_date", "view_volume"]
        ].rename(columns={"view_volume": "Previous_Views"})
        if pd.notna(coverage_scale):
            previous["Previous_Views"] *= coverage_scale
        previous["checkin_date"] = previous["checkin_date"].dt.normalize()
        funnel = funnel.merge(
            previous,
            on=["ProductID", "checkin_date"],
            how="left",
            validate="one_to_one",
        )
        previous_destination = build_latest_destination_demand(previous_demand).rename(
            columns={
                "Destination_Searches": "Previous_Destination_Searches",
                "Destination_Search_Snapshot": "Previous_Destination_Search_Snapshot",
            }
        )
        if pd.notna(coverage_scale):
            previous_destination["Previous_Destination_Searches"] *= coverage_scale
        previous_destination["checkin_date"] = previous_destination["checkin_date"].dt.normalize()
        funnel = funnel.merge(
            previous_destination,
            on=["Destination", "checkin_date"],
            how="left",
            validate="many_to_one",
        )
        new_vs_previous = funnel["Previous_Views"].isna() | funnel["check_status"].eq("new entry")
        funnel["Destination_Search_Change"] = (
            funnel["Destination_Searches"] - funnel["Previous_Destination_Searches"]
        )
        funnel["View_Change"] = funnel["view_volume"] - funnel["Previous_Views"].fillna(0)
        funnel["Upload_Change_Status"] = np.select(
            [
                new_vs_previous,
                funnel["View_Change"].gt(0),
                funnel["View_Change"].lt(0),
            ],
            ["New", "Up", "Down"],
            default="No change",
        )
    else:
        funnel["Previous_Views"] = np.nan
        funnel["Previous_Destination_Searches"] = np.nan
        funnel["Previous_Destination_Search_Snapshot"] = pd.NaT
        funnel["Destination_Search_Change"] = np.nan
        funnel["View_Change"] = np.nan
        funnel["Upload_Change_Status"] = "No baseline"
    return funnel


def build_demand_summary(demand: pd.DataFrame) -> pd.DataFrame:
    keys = ["ProductID", "checkin_date"]
    ordered = _with_observation_batches(demand).sort_values(keys + ["snapshot_at"]).copy()
    ordered = ordered.drop_duplicates(["Observation_Batch"] + keys, keep="last")
    ordered["Status_Modified"] = ordered["check_status"].eq("modify")

    interval_by_checkin = ordered.groupby(keys, as_index=False).agg(
        hotness_score=("hotness_score", "mean"),
        trend_momentum=("trend_momentum", "mean"),
        view_volume=("view_volume", "sum"),
        snapshot_at=("snapshot_at", "max"),
    )
    # ProductID is the stable hotel key. Hotel names can change between exports,
    # so select the newest name instead of grouping one ProductID into multiple
    # summary rows under its historical names.
    latest_names = (
        ordered.sort_values(["ProductID", "snapshot_at"])
        .drop_duplicates("ProductID", keep="last")
        [["ProductID", "ProductName"]]
    )
    current = (
        interval_by_checkin.groupby("ProductID", as_index=False)
        .agg(
            Current_Hotness=("hotness_score", "mean"),
            Peak_Hotness=("hotness_score", "max"),
            Avg_Hotness=("hotness_score", "mean"),
            Current_Trend=("trend_momentum", "mean"),
            Current_Views=("view_volume", "sum"),
            Peak_CheckIn_Views=("view_volume", "max"),
            Latest_Snapshot=("snapshot_at", "max"),
        )
    )
    movement = (
        ordered.groupby("ProductID", as_index=False)
        .agg(
            Active_View_Intervals=("view_volume", lambda values: int(values.gt(0).sum())),
            Modification_Count=("Status_Modified", "sum"),
        )
    )
    modified = (
        ordered.loc[ordered["Status_Modified"]]
        .groupby("ProductID", as_index=False)["snapshot_at"]
        .max()
        .rename(columns={"snapshot_at": "Last_Modified_At"})
    )
    peak_dates = (
        interval_by_checkin.sort_values(["ProductID", "view_volume", "checkin_date"])
        .drop_duplicates("ProductID", keep="last")
        [["ProductID", "checkin_date"]]
        .rename(columns={"checkin_date": "Peak_CheckIn_Date"})
    )
    summary = current.merge(latest_names, on="ProductID", validate="one_to_one")
    summary = summary.merge(movement, on="ProductID", validate="one_to_one")
    summary = summary.merge(modified, on="ProductID", how="left", validate="one_to_one")
    summary = summary.merge(peak_dates, on="ProductID", how="left", validate="one_to_one")
    summary["Demand_Score"] = (
        summary["Current_Hotness"].fillna(0) * 40
        + summary["Current_Trend"].fillna(0) * 20
        + _safe_max_score(summary["Current_Views"]) * 0.20
        + _safe_max_score(summary["Active_View_Intervals"]) * 0.20
    ).round(2)
    return summary


def build_engine(
    summary: pd.DataFrame,
    master: pd.DataFrame,
    performance: pd.DataFrame,
    funnel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    master = master.copy()
    performance = performance.copy()
    for frame in (master, performance):
        frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")

    master = master.sort_values("ProductID").drop_duplicates("ProductID", keep="last")
    raw_performance_columns = [
        "ProductID", "ProductName", "Agoda Status", "Ctrip Status", "Revenue", "B2B2C RN"
    ]
    performance = performance[raw_performance_columns]
    performance["Revenue"] = pd.to_numeric(performance["Revenue"], errors="coerce").fillna(0)
    performance["B2B2C RN"] = pd.to_numeric(performance["B2B2C RN"], errors="coerce").fillna(0)
    performance = (
        performance.sort_values(["ProductID", "Revenue", "B2B2C RN"])
        .drop_duplicates("ProductID", keep="last")
    )

    engine = summary.merge(
        master[["ProductID", "ProductName", "Destination", "HotelType Short", "Region"]],
        on="ProductID",
        how="left",
        suffixes=("", "_master"),
        validate="one_to_one",
    )
    engine["ProductName"] = engine["ProductName_master"].fillna(engine["ProductName"])
    engine = engine.drop(columns=["ProductName_master"])
    engine = engine.merge(
        performance.drop(columns=["ProductName"]),
        on="ProductID",
        how="left",
        validate="one_to_one",
    )
    if funnel is not None and not funnel.empty:
        hotel_funnel = (
            funnel.groupby("ProductID", as_index=False)
            .agg(
                Destination_Search_Context=("Destination_Searches", "max"),
                Funnel_Views=("view_volume", "sum"),
                Funnel_Bookings=("Bookings", "sum"),
                Internal_Worst_Gap=("Internal_Worst_Gap", "min"),
                Internal_Median_Gap=("Internal_Median_Gap", "median"),
                Internal_Comparable_Rates=("Internal_Comparable_Rates", "sum"),
                Internal_Total_Rates=("Internal_Total_Rates", "sum"),
                Agoda_Price_Disadvantage=("Agoda_Price_Disadvantage", "max"),
                Agoda_Comparable_Dates=("Agoda_Raw_Gap", "count"),
            )
        )
        hotel_funnel["Views_per_Search"] = np.where(
            hotel_funnel["Destination_Search_Context"].gt(0),
            hotel_funnel["Funnel_Views"] / hotel_funnel["Destination_Search_Context"],
            np.nan,
        )
        hotel_funnel["Search_to_Booking"] = np.where(
            hotel_funnel["Funnel_Views"].gt(0),
            hotel_funnel["Funnel_Bookings"] / hotel_funnel["Funnel_Views"],
            np.nan,
        )
        engine = engine.merge(hotel_funnel, on="ProductID", how="left", validate="one_to_one")

    engine["Agoda Status"] = _normalise_status(engine["Agoda Status"])
    engine["Ctrip Status"] = _normalise_status(engine["Ctrip Status"])
    engine["Revenue"] = engine["Revenue"].fillna(0)
    engine["B2B2C RN"] = engine["B2B2C RN"].fillna(0)
    engine["Revenue Score"] = _safe_max_score(engine["Revenue"]).round(2)
    engine["RN Score"] = _safe_max_score(engine["B2B2C RN"]).round(2)

    agoda_map = {"Mapped": 100, "No Room": 70, "Not Live": 40, "Not Mapped": 0, "Unknown": 0}
    ctrip_map = {"Mapped": 100, "Not Live": 40, "Not Mapped": 0, "Unknown": 0}
    engine["Agoda Score"] = engine["Agoda Status"].map(agoda_map).fillna(0)
    engine["Ctrip Score"] = engine["Ctrip Status"].map(ctrip_map).fillna(0)
    engine["Mapping Score"] = ((engine["Agoda Score"] + engine["Ctrip Score"]) / 2).round(2)
    engine["Business Score"] = (
        engine["Revenue Score"] * 0.5 + engine["RN Score"] * 0.3 + engine["Demand_Score"] * 0.2
    ).round(2)
    engine["Opportunity Index"] = (
        engine["Demand_Score"] * 0.40
        + engine["Revenue Score"] * 0.25
        + engine["RN Score"] * 0.15
        + engine["Current_Trend"].fillna(0).clip(0, 1) * 100 * 0.10
        + (100 - engine["Mapping Score"]) * 0.10
    ).round(2)

    engine = engine.sort_values("Opportunity Index", ascending=False).reset_index(drop=True)
    engine["Commercial Rank"] = np.arange(1, len(engine) + 1)
    engine["Priority"] = pd.cut(
        engine["Commercial Rank"],
        bins=[0, 50, 200, 500, np.inf],
        labels=["Critical", "High", "Medium", "Low"],
    ).astype(str)

    agoda = engine["Agoda Status"]
    ctrip = engine["Ctrip Status"]
    internal_gap = engine.get("Internal_Worst_Gap", pd.Series(np.nan, index=engine.index))
    agoda_disadvantage = engine.get(
        "Agoda_Price_Disadvantage", pd.Series(np.nan, index=engine.index)
    )
    views = engine.get("Funnel_Views", pd.Series(0, index=engine.index)).fillna(0)
    bookings = engine.get("Funnel_Bookings", pd.Series(0, index=engine.index)).fillna(0)
    conditions = [
        agoda.eq("Not Mapped"),
        agoda.eq("No Room"),
        agoda.eq("Not Live"),
        internal_gap.lt(0),
        agoda_disadvantage.gt(0),
        views.gt(0) & bookings.eq(0) & engine["Demand_Score"].ge(engine["Demand_Score"].median()),
        ctrip.eq("Not Mapped"),
        ctrip.eq("Not Live"),
        engine["Opportunity Index"].ge(70),
    ]
    engine["Action"] = np.select(
        conditions,
        [
            "Map Hotel to Agoda",
            "Complete Room-Type Mapping",
            "Activate Agoda",
            "Fix Internal Rate Parity",
            "Review Agoda Rate Competitiveness",
            "Investigate Booking Conversion",
            "Expand to Ctrip",
            "Activate Ctrip",
            "Protect Commercial Performance",
        ],
        default="Monitor",
    )
    reason_map = {
        "Map Hotel to Agoda": "Demand exists, but hotel-level mapping is missing on Agoda.",
        "Complete Room-Type Mapping": "Hotel-level mapping exists, but room types are not mapped.",
        "Activate Agoda": "Agoda connectivity exists, but the hotel is not live.",
        "Fix Internal Rate Parity": "Our rate is more expensive than a comparable partner rate.",
        "Review Agoda Rate Competitiveness": "Agoda reports a price disadvantage; 0% is cheapest.",
        "Investigate Booking Conversion": "Demand exists, but no booking is matched to the current check-in demand.",
        "Expand to Ctrip": "The hotel has demand but is not mapped on Ctrip.",
        "Activate Ctrip": "Ctrip connectivity exists, but the hotel is not live.",
        "Protect Commercial Performance": "This is a high-value hotel; monitor its commercial performance.",
        "Monitor": "No immediate distribution issue requires action.",
    }
    engine["Reason"] = engine["Action"].map(reason_map)
    return engine


def export_outputs(engine: pd.DataFrame, summary: pd.DataFrame, funnel: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DIR / "demand_summary.csv", index=False)
    engine.to_csv(PROCESSED_DIR / "engine.csv", index=False)
    funnel.to_csv(PROCESSED_DIR / "hotel_date_funnel.csv", index=False)
    engine.head(20).to_csv(OUTPUT_DIR / "top20.csv", index=False)
    engine[engine["Action"].eq("Map Hotel to Agoda")].to_csv(OUTPUT_DIR / "agoda_mapping.csv", index=False)
    engine[engine["Action"].eq("Expand to Ctrip")].to_csv(OUTPUT_DIR / "ctrip_mapping.csv", index=False)
    engine[engine["Action"].eq("Complete Room-Type Mapping")].to_csv(OUTPUT_DIR / "room_type_mapping.csv", index=False)


def run_pipeline(demand_path: Path | None = None):
    demand, master, performance, source = load_sources(demand_path)
    demand = prepare_demand(demand)
    previous_path = resolve_previous_demand_path()
    previous_demand = (
        prepare_demand(pd.read_csv(previous_path)) if previous_path is not None else None
    )
    current_period, baseline_period = select_interval_periods(demand, previous_demand)
    summary = build_demand_summary(current_period)
    funnel = build_hotel_date_funnel(current_period, baseline_period)
    engine = build_engine(summary, master, performance, funnel)
    export_outputs(engine, summary, funnel)
    return engine, demand, source
