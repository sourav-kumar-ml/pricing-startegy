# Pricing ML Project

Scripted, notebook-free pricing/elasticity workflow for café SKUs: EDA, feature engineering, normalization, time-aware splits, and baseline models.

## Repo Layout
- `data/raw/` – source CSVs + `CHECKSUMS.txt`
- `data/processed/` – generated features (`features.parquet`, `features_with_ids.parquet`), splits
- `configs/` – scaler, transform metadata, serialized models
- `reports/` – EDA outputs, correlations/VIFs, baseline metrics/coeffs, plots
- `src/eda/` – audit, collinearity, plotting
- `src/processing/` – ingestion + feature pipeline
- `src/models/` – splits, baseline models, elasticity, evaluation
- `tests/` – unit tests
- `docs/sprints/` – sprint summaries
- `docs/` – narrative reports, assumptions
- `scripts/` – pipeline runner
- `Plan.md` – project plan

## Setup
1) Python 3.14+; create/activate venv: `python -m venv .venv && source .venv/bin/activate`
2) Install deps: `pip install -r requirements.txt`
3) Optional: for headless plotting set `MPLBACKEND=Agg`; if `~/.matplotlib` isn’t writable set `MPLCONFIGDIR=$PWD/.cache/matplotlib`.

## Pipelines & Outputs
- **EDA (Sprint 1)**  
  Run: `MPLBACKEND=Agg .venv/bin/python -m src.eda.run_eda`  
  Outputs: `reports/eda_audit.md`, `reports/pearson_corr.csv`, `reports/spearman_corr.csv`, `reports/top_correlations.csv`, `reports/vif.csv`, plots under `reports/figures/`.
- **Transform pipeline (Sprint 2)**  
  Run: `.venv/bin/python -m src.processing.pipeline`  
  Outputs: `data/processed/features.parquet`, `data/processed/features_with_ids.parquet`, `configs/scaler.pkl`, `configs/transform_metadata.json`.
- **Time-aware splits (Sprint 3)**  
  Run: `.venv/bin/python -m src.models.splits`  
  Outputs: per-SKU train/val parquet files and `data/processed/splits/manifest.json`.
- **Baseline models (Sprint 3)**  
  Run: `.venv/bin/python -m src.models.baseline`  
  Outputs: `reports/baseline_metrics.csv`, `reports/baseline_coeffs.csv`, serialized models in `configs/models/`.
- **Elasticity & price curves (Sprint 4)**  
  Run: `.venv/bin/python -m src.models.elasticity`  
  Outputs: `reports/elasticity.csv`, `reports/price_curves.csv`, `reports/price_recommendations.csv`.
- **Evaluation & diagnostics (Sprint 5)**  
  Run: `.venv/bin/python -m src.models.eval`  
  Outputs: `reports/eval_metrics.csv`, `reports/eval_residuals.csv`, `reports/eval_rolling_cv.csv`.
- **End-to-end pipeline (Sprint 6)**  
  Run: `./scripts/pipeline.sh`  
  Executes: EDA → transform → splits → baseline → elasticity → evaluation (headless plotting).

## Tests
Run from repo root: `pytest` (or `.venv/bin/pytest`)

## CI
- GitHub Actions workflow (`.github/workflows/ci.yml`) runs pytest on push/PR (uses `ubuntu-latest`, Python 3.11).

## Status
- All sprints 1–6 complete; artifacts present under `data/processed/`, `configs/`, and `reports/`. Re-run pipelines if you need fresh outputs.

## Troubleshooting
- Matplotlib cache warning: set `MPLCONFIGDIR=$PWD/.cache/matplotlib`.
- Arrow sysctl warnings during Parquet ops in sandbox are harmless.
