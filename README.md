# Hotel Commercial Intelligence MVP

A Streamlit application that converts hotel-demand snapshots and commercial performance data into demand analytics, ranked opportunities, and hotel-level recommendations.

Search and view volumes are interval values. Near-identical crawler timestamps are consolidated into one observation batch. Destination search is represented once per `Observation Batch + Destination + CheckInDate` using the median reported value, then summed across intervals. Hotel views are represented once per `Observation Batch + ProductID + CheckInDate`, then summed across intervals. The engine uses destination search change for the market signal and hotel view level/change for hotel prioritization. `Continues`, `New Entry`, and `Modify` remain record-status metadata. `No Room` means hotel-level mapping is complete but room-type mapping is still required; it is not treated as an availability issue.

## Run

On macOS, double-click `run_app.command`. The first launch creates the virtual environment and installs dependencies.

Alternatively, run manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
streamlit run dashboard/Home.py
```

Open `http://localhost:8501`.

## Daily refresh

Use **Update demand data** on the Home page to upload only the newest daily demand export. HCI keeps `demand_latest.csv.gz` as the complete event history, saves the state before the upload as `demand_previous.csv.gz`, appends new events, removes duplicate `Time Stamp + CheckInDate + ProductID` keys, and rebuilds the dashboard. Uploading the same daily file twice is a safe no-op.

## Pages

- **Home** — portfolio health and top actions.
- **Demand Analytics** — demand trends, destinations, regions, and hotel ranking.
- **Opportunity Center** — commercial worklist with filters and export.
- **Hotel Explorer** — demand history and a complete hotel commercial profile.

The raw master and performance workbooks remain under `data/raw/`. BigQuery can later replace the CSV loader without changing the page logic.
# Hotel Commercial Intelligence Platform

## Run

From this folder:

```bash
.venv/bin/python main.py
.venv/bin/streamlit run dashboard/Home.py
```

The pipeline reads these source files from `data/raw/`:

- `demand_latest.csv.gz` (recommended for large files) or `demand_latest.csv`
- `demand_previous.csv.gz` (retained baseline for previous-versus-latest comparison)
- `Booking_Production.csv`
- `Hotel_Performance.xlsx`
- `Master_Hotel.xlsx`
- `internal price gap.csv`
- `agoda price gap.xlsx`

It produces:

- `data/processed/demand_summary.csv`
- `data/processed/hotel_date_funnel.csv`
- `data/processed/engine.csv`

Demand sums interval observations for destination Search and hotel View. Previous-period values are normalized to the same number of observation batches when coverage differs. Bookings join on the hotel/date key. Internal and Agoda
parity are kept as separate signals and matched by the same calendar date,
which the dashboard labels as first-night parity.
