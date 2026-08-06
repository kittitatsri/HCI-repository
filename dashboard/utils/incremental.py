"""Incremental demand ingestion kept inside the dashboard package for cloud-safe imports."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
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
VALID_CHECK_STATUSES = {"continues", "new entry", "modify"}


def _parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str.replace("\u202f", " ", regex=False),
        format="mixed",
        errors="coerce",
    )


def merge_incremental_demand(
    history: pd.DataFrame, incremental: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int | bool | pd.Timestamp]]:
    """Append one incremental export to demand history without double-counting events."""
    for label, frame in (("Current history", history), ("Uploaded file", incremental)):
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {', '.join(sorted(missing))}")

    incoming = incremental.copy()
    incoming_status = incoming["check_status"].fillna("").astype(str).str.strip().str.lower()
    invalid_statuses = sorted(
        set(incoming_status[incoming_status.ne("")]).difference(VALID_CHECK_STATUSES)
    )
    if invalid_statuses:
        raise ValueError(
            "Uploaded check_status contains unsupported values: " + ", ".join(invalid_statuses)
        )

    combined = pd.concat([history, incoming], ignore_index=True, sort=False)
    combined["_snapshot_key"] = _parse_timestamp(combined["Time Stamp"])
    combined["_checkin_key"] = pd.to_datetime(
        combined["CheckInDate"], errors="coerce"
    ).dt.normalize()
    combined["_product_key"] = pd.to_numeric(
        combined["ProductID"], errors="coerce"
    ).astype("Int64")

    invalid_key = combined[["_snapshot_key", "_checkin_key", "_product_key"]].isna().any(axis=1)
    history_invalid = int(invalid_key.iloc[: len(history)].sum())
    upload_invalid = int(invalid_key.iloc[len(history) :].sum())
    if upload_invalid:
        raise ValueError(
            f"The uploaded file contains {upload_invalid:,} rows with an invalid Time Stamp, "
            "CheckInDate, or ProductID. Correct these rows before uploading."
        )

    combined["_row_fallback"] = 0
    if history_invalid:
        combined.loc[invalid_key, "_row_fallback"] = combined.index[invalid_key] + 1
    event_keys = ["_snapshot_key", "_checkin_key", "_product_key", "_row_fallback"]
    helper_columns = event_keys

    history_clean = (
        combined.iloc[: len(history)]
        .drop_duplicates(event_keys, keep="last")
        .sort_values("_snapshot_key", kind="stable")
    )
    before_dedup = len(combined)
    combined = (
        combined.drop_duplicates(event_keys, keep="last")
        .sort_values("_snapshot_key", kind="stable")
    )

    history_keys = pd.MultiIndex.from_arrays(
        [
            _parse_timestamp(history["Time Stamp"]),
            pd.to_datetime(history["CheckInDate"], errors="coerce").dt.normalize(),
            pd.to_numeric(history["ProductID"], errors="coerce").astype("Int64"),
        ]
    )
    incoming_keys = pd.MultiIndex.from_arrays(
        [
            _parse_timestamp(incoming["Time Stamp"]),
            pd.to_datetime(incoming["CheckInDate"], errors="coerce").dt.normalize(),
            pd.to_numeric(incoming["ProductID"], errors="coerce").astype("Int64"),
        ]
    )
    new_events = int((~incoming_keys.unique().isin(history_keys)).sum())

    result = combined.drop(columns=helper_columns)
    result = result.reindex(
        columns=list(history.columns) + [column for column in result.columns if column not in history.columns]
    )
    comparable_history = history_clean.drop(columns=helper_columns).reindex(columns=result.columns)
    data_changed = not result.reset_index(drop=True).equals(
        comparable_history.reset_index(drop=True)
    )
    return result, {
        "history_rows": len(history),
        "uploaded_rows": len(incremental),
        "new_events": new_events,
        "duplicates_removed": before_dedup - len(combined),
        "merged_rows": len(result),
        "data_changed": data_changed,
        "upload_min_timestamp": _parse_timestamp(incoming["Time Stamp"]).min(),
        "upload_max_timestamp": _parse_timestamp(incoming["Time Stamp"]).max(),
    }
