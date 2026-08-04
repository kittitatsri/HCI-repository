# Hotel Commercial Intelligence MVP

A Streamlit application that converts hotel-demand snapshots and commercial performance data into demand analytics, ranked opportunities, and hotel-level recommendations.

Search and view volumes are treated as persistent cumulative counters. The engine keeps current cumulative totals and separately calculates observed positive increases between snapshots. `No Room` means hotel-level mapping is complete but room-type mapping is still required; it is not treated as an availability issue.

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

Use **Update demand data** on the Home page to upload the latest demand CSV. The app validates the columns, rebuilds `demand_summary.csv` and `engine.csv`, and refreshes the dashboards.

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

- `demand_latest.csv`
- `Booking_Production.csv`
- `Hotel_Performance.xlsx`
- `Master_Hotel.xlsx`
- `internal price gap.csv`
- `agoda price gap.xlsx`

It produces:

- `data/processed/demand_summary.csv`
- `data/processed/hotel_date_funnel.csv`
- `data/processed/engine.csv`

Demand uses the latest cumulative Search and View value for each
`ProductID + CheckInDate`. Bookings join on the same key. Internal and Agoda
parity are kept as separate signals and matched by the same calendar date,
which the dashboard labels as first-night parity.
