"""Read-only reconciliation checks for HCI demand metrics.

Run from the repository root with:
    .venv/bin/python scripts/audit_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline import (  # noqa: E402
    _with_observation_batches,
    build_latest_demand_by_checkin,
    build_latest_destination_demand,
    build_stored_history_outputs,
    prepare_demand,
    resolve_demand_path,
    resolve_previous_demand_path,
    select_interval_periods,
)


def _raw_key_quality(frame: pd.DataFrame) -> dict[str, int]:
    timestamps = pd.to_datetime(
        frame["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False),
        format="mixed",
        errors="coerce",
    )
    checkins = pd.to_datetime(frame["CheckInDate"], errors="coerce").dt.normalize()
    products = pd.to_numeric(frame["ProductID"], errors="coerce")
    keys = pd.DataFrame({"timestamp": timestamps, "checkin": checkins, "product": products})
    return {
        "invalid_timestamp_rows": int(timestamps.isna().sum()),
        "invalid_checkin_rows": int(checkins.isna().sum()),
        "missing_product_rows": int(products.isna().sum()),
        "duplicate_event_keys": int(keys.duplicated().sum()),
    }


def _master_conflicts(master: pd.DataFrame) -> tuple[int, int]:
    frame = master.copy()
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    columns = [
        column
        for column in [
            "ProductName", "Destination", "Region", "HotelType Short", "Agoda Status", "Ctrip Status"
        ]
        if column in frame
    ]
    duplicated = frame[frame["ProductID"].duplicated(False) & frame["ProductID"].notna()]
    conflict_ids = 0
    for _, group in duplicated.groupby("ProductID"):
        if any(group[column].fillna("<NA>").astype(str).nunique() > 1 for column in columns):
            conflict_ids += 1
    return int(duplicated["ProductID"].nunique()), conflict_ids


def main() -> int:
    latest_raw = pd.read_csv(resolve_demand_path())
    previous_path = resolve_previous_demand_path()
    if previous_path is None:
        raise FileNotFoundError("demand_previous.csv.gz or demand_previous.csv is required")
    previous_raw = pd.read_csv(previous_path)
    latest = prepare_demand(latest_raw)
    previous = prepare_demand(previous_raw)
    current, baseline = select_interval_periods(latest, previous)
    expected_history_hotels, expected_history_destinations = build_stored_history_outputs(
        latest, previous
    )

    current_batches = _with_observation_batches(current)["Observation_Batch"].nunique()
    baseline_batches = _with_observation_batches(baseline)["Observation_Batch"].nunique()
    hard_failures: list[str] = []
    if current_batches != baseline_batches:
        hard_failures.append(
            f"Comparison windows differ: current={current_batches}, baseline={baseline_batches}"
        )

    expected_destinations = build_latest_destination_demand(current)
    expected_hotels = build_latest_demand_by_checkin(current)
    baseline_destinations = build_latest_destination_demand(baseline).rename(
        columns={
            "Destination_Searches": "Previous_Destination_Searches",
            "Destination_Intervals": "Previous_Destination_Intervals",
        }
    )
    expected_destinations = expected_destinations.merge(
        baseline_destinations[
            [
                "Destination", "checkin_date", "Previous_Destination_Searches",
                "Previous_Destination_Intervals",
            ]
        ],
        on=["Destination", "checkin_date"],
        how="left",
        validate="one_to_one",
    )
    expected_destinations["Comparable_Destination_Intervals"] = np.minimum(
        expected_destinations["Destination_Intervals"],
        expected_destinations["Previous_Destination_Intervals"],
    )
    expected_destinations["Comparable_Destination_Searches"] = (
        expected_destinations["Destination_Searches"]
        / expected_destinations["Destination_Intervals"]
        * expected_destinations["Comparable_Destination_Intervals"]
    )
    expected_destinations["Previous_Comparable_Destination_Searches"] = (
        expected_destinations["Previous_Destination_Searches"]
        / expected_destinations["Previous_Destination_Intervals"]
        * expected_destinations["Comparable_Destination_Intervals"]
    )
    baseline_hotels = build_latest_demand_by_checkin(baseline).rename(
        columns={"view_volume": "Previous_Views", "Interval_Count": "Previous_View_Intervals"}
    )
    expected_hotels = expected_hotels.merge(
        baseline_hotels[
            ["ProductID", "checkin_date", "Previous_Views", "Previous_View_Intervals"]
        ],
        on=["ProductID", "checkin_date"],
        how="left",
        validate="one_to_one",
    )
    expected_hotels["Comparable_View_Intervals"] = np.minimum(
        expected_hotels["Interval_Count"], expected_hotels["Previous_View_Intervals"]
    )
    expected_hotels["Comparable_Views"] = (
        expected_hotels["view_volume"] / expected_hotels["Interval_Count"]
        * expected_hotels["Comparable_View_Intervals"]
    )
    expected_hotels["Previous_Comparable_Views"] = (
        expected_hotels["Previous_Views"] / expected_hotels["Previous_View_Intervals"]
        * expected_hotels["Comparable_View_Intervals"]
    )
    processed = pd.read_csv(ROOT / "data" / "processed" / "hotel_date_funnel.csv")
    processed["ProductID"] = pd.to_numeric(processed["ProductID"], errors="coerce").astype("Int64")
    processed["checkin_date"] = pd.to_datetime(processed["checkin_date"], errors="coerce").dt.normalize()

    destination_value_columns = [
        "Destination_Searches", "Comparable_Destination_Searches",
        "Previous_Comparable_Destination_Searches",
    ]
    actual_destinations = processed.drop_duplicates(["Destination", "checkin_date"])[
        ["Destination", "checkin_date"] + destination_value_columns
    ]
    destination_check = expected_destinations.merge(
        actual_destinations,
        on=["Destination", "checkin_date"],
        how="outer",
        suffixes=("_expected", "_actual"),
        indicator=True,
    )
    destination_differences = [
        (
            destination_check[f"{column}_expected"]
            - destination_check[f"{column}_actual"]
        ).abs().fillna(0)
        for column in destination_value_columns
    ]
    destination_check["difference"] = pd.concat(destination_differences, axis=1).max(axis=1)
    destination_mismatches = int(
        destination_check["_merge"].ne("both").sum()
        + destination_check["difference"].fillna(np.inf).gt(1e-6).sum()
    )
    if destination_mismatches:
        hard_failures.append(f"Destination-search reconciliation mismatches: {destination_mismatches}")

    hotel_value_columns = ["view_volume", "Comparable_Views", "Previous_Comparable_Views"]
    actual_hotels = processed[["ProductID", "checkin_date"] + hotel_value_columns]
    hotel_check = expected_hotels[["ProductID", "checkin_date"] + hotel_value_columns].merge(
        actual_hotels,
        on=["ProductID", "checkin_date"],
        how="outer",
        suffixes=("_expected", "_actual"),
        indicator=True,
    )
    hotel_differences = [
        (hotel_check[f"{column}_expected"] - hotel_check[f"{column}_actual"]).abs().fillna(0)
        for column in hotel_value_columns
    ]
    hotel_check["difference"] = pd.concat(hotel_differences, axis=1).max(axis=1)
    hotel_mismatches = int(
        hotel_check["_merge"].ne("both").sum()
        + hotel_check["difference"].fillna(np.inf).gt(1e-6).sum()
    )
    if hotel_mismatches:
        hard_failures.append(f"Hotel-view reconciliation mismatches: {hotel_mismatches}")

    stored_history_destinations = pd.read_csv(
        ROOT / "data" / "processed" / "historical_destination_date.csv"
    )
    stored_history_destinations["checkin_date"] = pd.to_datetime(
        stored_history_destinations["checkin_date"], errors="coerce"
    ).dt.normalize()
    history_destination_check = expected_history_destinations[
        ["Destination", "checkin_date", "Total_Observed_Searches"]
    ].merge(
        stored_history_destinations[
            ["Destination", "checkin_date", "Total_Observed_Searches"]
        ],
        on=["Destination", "checkin_date"], how="outer",
        suffixes=("_expected", "_actual"), indicator=True,
    )
    history_destination_mismatches = int(
        history_destination_check["_merge"].ne("both").sum()
        + (
            history_destination_check["Total_Observed_Searches_expected"]
            - history_destination_check["Total_Observed_Searches_actual"]
        ).abs().fillna(np.inf).gt(1e-6).sum()
    )
    if history_destination_mismatches:
        hard_failures.append(
            f"Historical destination mismatches: {history_destination_mismatches}"
        )
    stored_history_hotels = pd.read_csv(
        ROOT / "data" / "processed" / "historical_hotel_date.csv"
    )
    for frame in (expected_history_hotels, stored_history_hotels):
        frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
        frame["checkin_date"] = pd.to_datetime(
            frame["checkin_date"], errors="coerce"
        ).dt.normalize()
    history_hotel_check = expected_history_hotels[
        ["ProductID", "checkin_date", "Total_Observed_Views"]
    ].merge(
        stored_history_hotels[["ProductID", "checkin_date", "Total_Observed_Views"]],
        on=["ProductID", "checkin_date"], how="outer",
        suffixes=("_expected", "_actual"), indicator=True,
    )
    history_hotel_mismatches = int(
        history_hotel_check["_merge"].ne("both").sum()
        + (
            history_hotel_check["Total_Observed_Views_expected"]
            - history_hotel_check["Total_Observed_Views_actual"]
        ).abs().fillna(np.inf).gt(1e-6).sum()
    )
    if history_hotel_mismatches:
        hard_failures.append(f"Historical hotel mismatches: {history_hotel_mismatches}")

    duplicate_master_ids, conflicting_master_ids = _master_conflicts(
        pd.read_excel(ROOT / "data" / "raw" / "Master_Hotel.xlsx")
    )
    current_batched = _with_observation_batches(current)
    search_groups = current_batched.groupby(
        ["Observation_Batch", "Destination", "checkin_date"]
    )["search_volume"].nunique()
    inconsistent_search_groups = int(search_groups.gt(1).sum())
    batch_rows = current_batched.groupby("Observation_Batch").size()
    print("HCI demand audit")
    print(f"Current window: {current.snapshot_at.min()} to {current.snapshot_at.max()}")
    print(f"Baseline window: {baseline.snapshot_at.min()} to {baseline.snapshot_at.max()}")
    print(f"Observation batches: current={current_batches}, baseline={baseline_batches}")
    print(f"Latest raw quality: {_raw_key_quality(latest_raw)}")
    print(f"Previous raw quality: {_raw_key_quality(previous_raw)}")
    print(f"Processed destination mismatches: {destination_mismatches}")
    print(f"Processed hotel mismatches: {hotel_mismatches}")
    print(f"Historical destination mismatches: {history_destination_mismatches}")
    print(f"Historical hotel mismatches: {history_hotel_mismatches}")
    audit_date = pd.Timestamp("2026-08-19")
    audit_total = expected_history_destinations.loc[
        expected_history_destinations["checkin_date"].eq(audit_date),
        "Total_Observed_Searches",
    ].sum()
    print(f"19 Aug total observed searches across stored history: {audit_total:,.1f}")
    print(
        f"Search groups with conflicting raw values: {inconsistent_search_groups:,} "
        "(pipeline uses the median per destination interval)"
    )
    print(
        "Rows per observation batch: "
        f"min={batch_rows.min():,}, median={batch_rows.median():,.0f}, max={batch_rows.max():,}"
    )
    print(
        f"Master duplicate Product IDs: {duplicate_master_ids}; "
        f"conflicting duplicate IDs: {conflicting_master_ids}"
    )
    if hard_failures:
        print("FAIL")
        for failure in hard_failures:
            print(f"- {failure}")
        return 1
    print("PASS — processed demand values reconcile to the selected raw interval window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
