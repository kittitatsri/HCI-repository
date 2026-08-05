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
def _read_engine(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def load_engine() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "engine.csv"
    return _read_engine(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def _read_demand(path: str, modified_ns: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["snapshot_at"] = pd.to_datetime(
        frame["Time Stamp"].astype(str).str.replace("\u202f", " ", regex=False), errors="coerce"
    )
    frame["checkin_date"] = pd.to_datetime(frame["CheckInDate"], errors="coerce")
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    return frame


def load_demand() -> pd.DataFrame:
    path = demand_source_path()
    return _read_demand(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def _read_funnel(path: str, modified_ns: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["checkin_date"] = pd.to_datetime(frame["checkin_date"], errors="coerce")
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce")
    frame["ProductID"] = pd.to_numeric(frame["ProductID"], errors="coerce").astype("Int64")
    return frame


def load_funnel() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "hotel_date_funnel.csv"
    return _read_funnel(str(path), path.stat().st_mtime_ns)


def clear_data_cache() -> None:
    _read_engine.clear()
    _read_demand.clear()
    _read_funnel.clear()
