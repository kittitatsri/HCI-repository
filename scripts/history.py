from pathlib import Path
from datetime import datetime

def save_snapshot(engine):

    folder = Path("data/processed/snapshots")

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = datetime.today().strftime(
        "engine_%Y%m%d.csv"
    )

    engine.to_csv(
        folder / filename,
        index=False
    )

    print(f"✅ Snapshot saved: {filename}")