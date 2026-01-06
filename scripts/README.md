# Scripts

## pipeline.sh
- Runs the end-to-end workflow (EDA → transform → splits → baseline models → elasticity → evaluation).
- Usage: `./scripts/pipeline.sh`
- Requires: Python 3.14+, `.venv` with `pip install -r requirements.txt`.
- Notes: sets `MPLBACKEND=Agg` for headless plotting.
