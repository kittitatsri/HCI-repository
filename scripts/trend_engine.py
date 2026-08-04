import pandas as pd
from pathlib import Path

def latest_snapshots():

    folder = Path("data/processed/snapshots")

    files = sorted(folder.glob("engine_*.csv"))

    if len(files) < 2:
        return None, None

    yesterday = pd.read_csv(files[-2])

    today = pd.read_csv(files[-1])

    return yesterday, today