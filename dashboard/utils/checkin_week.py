from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


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
