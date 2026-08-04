from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def demand_source_path() -> Path:
    for filename in ("demand_latest.csv.gz", "demand_latest.csv"):
        path = ROOT / "data" / "raw" / filename
        if path.exists():
            return path
    raise FileNotFoundError("Missing demand_latest.csv.gz or demand_latest.csv")


@st.cache_data(show_spinner=False)
def load_engine() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "engine.csv")


@st.cache_data(show_spinner=False)
def load_demand() -> pd.DataFrame:
    frame = pd.read_csv(demand_source_path())
    frame["snapshot_at"] = pd.to_datetime(
        frame["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False), errors="coerce"
    )
    frame["checkin_date"] = pd.to_datetime(frame["CheckInDate"], errors="coerce")
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    return frame


@st.cache_data(show_spinner=False)
def load_funnel() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data" / "processed" / "hotel_date_funnel.csv")
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce")
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce")
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    return frame


def clear_data_cache() -> None:
    load_engine.clear()
    load_demand.clear()
    load_funnel.clear()
