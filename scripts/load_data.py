from pathlib import Path
import pandas as pd


def load_data():
    # Project root
    BASE_DIR = Path(__file__).resolve().parent.parent

    RAW_DIR = BASE_DIR / "data" / "raw"

    # Automatically find the newest CSV
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV found in data/raw")

    demand_file = csv_files[-1]
    master_file = RAW_DIR / "Master_Hotel.xlsx"
    performance_file = RAW_DIR / "Hotel_Performance.xlsx"

    demand = pd.read_csv(demand_file)
    master = pd.read_excel(master_file)
    performance = pd.read_excel(performance_file)

    return demand, master, performance