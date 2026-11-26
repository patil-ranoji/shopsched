# ShopSched 2.0

Manufacturing production planning and scheduling application built with Streamlit, pandas, Plotly and OR-Tools.

## Included
- FCFS, SPT, EDD and CP-SAT scheduling
- Batch-aware routing: production quantity is split into configured batch sizes
- Interactive job, routing and machine tables plus add forms
- Machine register with shift definitions and active/inactive status
- Breakdown and planned-maintenance downtime records
- CP-SAT machine non-overlap and downtime constraints
- Shop-floor operator start/pause/complete and event tracking
- Excel import/export and CSV schedule export
- PDF schedule report and machine-load summary
- SQLite for local development; PostgreSQL via `DATABASE_URL` for persistent hosting

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to your GitHub repository.
2. Create or update the Streamlit Cloud app to use `app.py`.
3. For persistent data, create a managed PostgreSQL database (e.g. Supabase or Neon).
4. In Streamlit Cloud: app → Settings → Secrets, add:
   ```toml
   DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE"
   ```
5. Reboot the app. Keep the database URL private; never commit it to GitHub.

## Important limitations in this release
- Machine shift start/end and working days are stored and editable, but the current scheduling algorithms do not yet constrain operations to those shift windows. CP-SAT does enforce machine non-overlap and the recorded downtime intervals.
- Operator start/pause/complete events are recorded in the database. Operator status and the in-memory current schedule are held in Streamlit session state; generate a new schedule after downtime/production changes.
- Times are modelled as elapsed hours from a common schedule origin, not calendar timestamps.
- SQLite is suitable for local experimentation. Streamlit Community Cloud local filesystems are ephemeral; use PostgreSQL for persistent cloud data.
- Review all plans before using them for actual production. This tool is an educational/decision-support prototype, not a certified MES.

## Excel format
The exported workbook includes `Jobs`, `Operations`, `Machines`, and `Downtime` sheets. Import expects at least `Jobs` and `Operations` with their exported column headers.
