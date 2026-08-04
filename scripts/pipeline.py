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


def resolve_demand_path() -> Path:
    for filename in DEMAND_FILENAMES:
        path = RAW_DIR / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing demand source. Add {DEMAND_FILENAMES[0]} or {DEMAND_FILENAMES[1]} to {RAW_DIR}"
    )


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
        errors="coerce",
    )
    demand["checkin_date"] = pd.to_datetime(demand["CheckInDate"], errors="coerce")
    for column in ["trend_momentum", "hotness_score", "search_volume", "view_volume"]:
        demand[column] = pd.to_numeric(demand[column], errors="coerce")
    demand["check_status"] = demand["check_status"].fillna("unknown").astype(str).str.strip().str.lower()
    return demand.dropna(subset=["ProductID", "ProductName"])


def build_latest_demand_by_checkin(demand: pd.DataFrame) -> pd.DataFrame:
    """Return one current cumulative demand record per hotel/check-in date."""
    keys = ["ProductID", "checkin_date"]
    latest = (
        demand.dropna(subset=keys + ["snapshot_at"])
        .sort_values(keys + ["snapshot_at"])
        .drop_duplicates(keys, keep="last")
        .copy()
    )
    return latest[
        [
            "ProductID",
            "ProductName",
            "Destination",
            "checkin_date",
            "snapshot_at",
            "hotness_score",
            "trend_momentum",
            "search_volume",
            "view_volume",
            "check_status",
        ]
    ]


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


def build_hotel_date_funnel(demand: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_demand_by_checkin(demand)
    latest["checkin_date"] = latest["checkin_date"].dt.normalize()
    funnel = latest.merge(
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
        funnel["search_volume"].gt(0),
        funnel["view_volume"] / funnel["search_volume"],
        np.nan,
    )
    funnel["Search_to_Booking"] = np.where(
        funnel["search_volume"].gt(0),
        funnel["Bookings"] / funnel["search_volume"],
        np.nan,
    )
    funnel["Internal_Comparable_Coverage"] = np.where(
        funnel["Internal_Total_Rates"].gt(0),
        funnel["Internal_Comparable_Rates"] / funnel["Internal_Total_Rates"],
        np.nan,
    )
    return funnel


def build_demand_summary(demand: pd.DataFrame) -> pd.DataFrame:
    keys = ["ProductID", "CheckInDate"]
    ordered = demand.sort_values(keys + ["snapshot_at"]).copy()

    # Search and view are persistent cumulative counters. Their raw values
    # describe the current total, while positive differences describe new
    # activity observed within the loaded period.
    ordered["Search_Increment"] = (
        ordered.groupby(keys)["search_volume"].diff().fillna(0).clip(lower=0)
    )
    ordered["View_Increment"] = (
        ordered.groupby(keys)["view_volume"].diff().fillna(0).clip(lower=0)
    )
    ordered["Status_Modified"] = ordered["check_status"].eq("modify")

    latest_by_checkin = ordered.drop_duplicates(keys, keep="last")
    current = (
        latest_by_checkin.groupby(["ProductID", "ProductName"], as_index=False)
        .agg(
            Current_Hotness=("hotness_score", "mean"),
            Peak_Hotness=("hotness_score", "max"),
            Avg_Hotness=("hotness_score", "mean"),
            Current_Trend=("trend_momentum", "mean"),
            Current_Search=("search_volume", "sum"),
            Current_Views=("view_volume", "sum"),
            Peak_CheckIn_Search=("search_volume", "max"),
            Peak_CheckIn_Views=("view_volume", "max"),
            Latest_Snapshot=("snapshot_at", "max"),
        )
    )
    movement = (
        ordered.groupby(["ProductID", "ProductName"], as_index=False)
        .agg(
            Observed_Search_Increase=("Search_Increment", "sum"),
            Observed_View_Increase=("View_Increment", "sum"),
            Modification_Count=("Status_Modified", "sum"),
        )
    )
    modified = (
        ordered.loc[ordered["Status_Modified"]]
        .groupby(["ProductID", "ProductName"], as_index=False)["snapshot_at"]
        .max()
        .rename(columns={"snapshot_at": "Last_Modified_At"})
    )
    peak_dates = (
        latest_by_checkin.sort_values(["ProductID", "search_volume", "checkin_date"])
        .drop_duplicates("ProductID", keep="last")
        [["ProductID", "checkin_date"]]
        .rename(columns={"checkin_date": "Peak_CheckIn_Date"})
    )
    summary = current.merge(movement, on=["ProductID", "ProductName"], validate="one_to_one")
    summary = summary.merge(modified, on=["ProductID", "ProductName"], how="left", validate="one_to_one")
    summary = summary.merge(peak_dates, on="ProductID", how="left", validate="one_to_one")
    summary["Demand_Score"] = (
        summary["Current_Hotness"].fillna(0) * 40
        + summary["Current_Trend"].fillna(0) * 20
        + _safe_max_score(summary["Observed_Search_Increase"]) * 0.25
        + _safe_max_score(summary["Observed_View_Increase"]) * 0.15
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
                Funnel_Searches=("search_volume", "sum"),
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
            hotel_funnel["Funnel_Searches"].gt(0),
            hotel_funnel["Funnel_Views"] / hotel_funnel["Funnel_Searches"],
            np.nan,
        )
        hotel_funnel["Search_to_Booking"] = np.where(
            hotel_funnel["Funnel_Searches"].gt(0),
            hotel_funnel["Funnel_Bookings"] / hotel_funnel["Funnel_Searches"],
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
    searches = engine.get("Funnel_Searches", pd.Series(0, index=engine.index)).fillna(0)
    bookings = engine.get("Funnel_Bookings", pd.Series(0, index=engine.index)).fillna(0)
    conditions = [
        agoda.eq("Not Mapped"),
        agoda.eq("No Room"),
        agoda.eq("Not Live"),
        internal_gap.lt(0),
        agoda_disadvantage.gt(0),
        searches.gt(0) & bookings.eq(0) & engine["Demand_Score"].ge(engine["Demand_Score"].median()),
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
    summary = build_demand_summary(demand)
    funnel = build_hotel_date_funnel(demand)
    engine = build_engine(summary, master, performance, funnel)
    export_outputs(engine, summary, funnel)
    return engine, demand, source
