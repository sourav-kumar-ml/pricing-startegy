# Pricing ML Project

End-to-end pricing/elasticity workflow for café SKUs with scripted EDA, feature engineering, normalization, and time-aware splits (no notebooks).

## Repo Layout
- `data/raw/`: source CSVs + `CHECKSUMS.txt`
- `data/processed/`: generated features and split datasets
- `configs/`: scaler + transform metadata
- `src/eda/`: audit, collinearity, and plotting scripts
- `src/processing/`: ingestion + feature pipeline
- `src/models/`: split utilities (and upcoming models)
- `tests/`: unit tests
- `docs/sprints/`: sprint summaries
- `Plan.md`: project plan and sprint tasks

## Setup
1) Use Python 3.14+ and a local venv:  
   `python -m venv .venv && source .venv/bin/activate`
2) Install dependencies:  
   `pip install -r requirements.txt`
3) (Optional) For headless plotting, set `MPLBACKEND=Agg`. If `~/.matplotlib` isn’t writable, set `MPLCONFIGDIR=$PWD/.cache/matplotlib`.

## Running Pipelines
- **EDA (Sprint 1 artifacts):**  
  `MPLBACKEND=Agg .venv/bin/python -m src.eda.run_eda`  
  Outputs to `reports/` (audit, correlations, VIFs, plots).
- **Transform pipeline (Sprint 2 artifacts):**  
  `.venv/bin/python -m src.processing.pipeline`  
  Outputs: `data/processed/features.parquet`, `data/processed/features_with_ids.parquet`, `configs/scaler.pkl`, `configs/transform_metadata.json`.
- **Time-aware splits (Sprint 3 step 1):**  
  `.venv/bin/python -m src.models.splits`  
  Outputs: per-SKU train/val parquet files and `data/processed/splits/manifest.json`.

## Tests
Run the full suite from repo root:  
`pytest` (or `.venv/bin/pytest`)

## Next Steps
- Implement baseline per-SKU models (OLS + regularized) under `src/models/` and document in `docs/sprints/`.
