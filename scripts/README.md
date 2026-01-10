# Scripts

## pipeline.sh
- Runs the end-to-end workflow (EDA → transform → splits → baseline models → elasticity → evaluation).
- Usage: `./scripts/pipeline.sh`
- Requires: Python 3.11+, `.venv` with `pip install -r requirements.txt`.
- Notes: sets `MPLBACKEND=Agg` for headless plotting.

## gen_import_map.py
- Generates an internal import graph for the repo.
- Outputs: `docs/IMPORT_MAP.md`, `docs/IMPORT_MAP.dot`
- Usage: `python3 scripts/gen_import_map.py`
