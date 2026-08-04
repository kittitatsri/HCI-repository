from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_engine() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "engine.csv")


@st.cache_data(show_spinner=False)
def load_demand() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "data" / "raw" / "demand_latest.csv")
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
