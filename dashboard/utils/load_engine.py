import pandas as pd
import streamlit as st


@st.cache_data
def load_engine():

    return pd.read_csv(
        "data/processed/engine.csv"
    )