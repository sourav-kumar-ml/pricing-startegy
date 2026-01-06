# Sprint 3 Summary - Modeling Readiness & Hygiene (Status: Completed)

## Scope and Intent
- Produce time-aware train/validation splits per SKU and baseline elasticities models (OLS + Ridge).
- Persist model metrics/coefs and artifacts for downstream evaluation.

## Work Completed

### 1) ID-aware features and splits
- What: Pipeline now saves `features_with_ids.parquet` (includes `SELL_ID`, `CALENDAR_DATE`, target, features) to support chronological splits.
- Why: Needed to split per SKU without losing identifiers.
- How: Extended `src/processing/pipeline.py` to emit the ID-aware dataset and record its path in `transform_metadata.json`.
- Result: `data/processed/features_with_ids.parquet`.

### 2) Time-based per-SKU splits
- What: Chronological 80/20 splits per `SELL_ID`, plus manifest with row counts and date ranges.
- Why: Prevent temporal leakage and prep train/val sets for modeling.
- How: `src/models/splits.py` (`time_based_split`, `split_by_sku`, `save_manifest`).
- Result: `data/processed/splits/*_train.parquet`, `*_val.parquet`, manifest at `data/processed/splits/manifest.json`.

### 3) Baseline models and metrics
- What: Fit per-SKU OLS and Ridge models on transformed features; saved metrics, coefficients, and serialized models.
- Why: Provide a first-cut elasticity baseline and a template for future models.
- How: `src/models/baseline.py` loads splits, trains models, computes RMSE on train/val, saves metrics/coefs to `reports/`, and models to `configs/models/`.
- Result: `reports/baseline_metrics.csv`, `reports/baseline_coeffs.csv`, serialized models in `configs/models/`.

### 4) Testing
- What: Added tests for baseline feature selection and training flow.
- Why: Guard against regressions in modeling utilities.
- How: `tests/test_baseline.py`; full suite green.

## Artifacts Produced
- Data: `data/processed/features_with_ids.parquet`, `data/processed/splits/*`, `data/processed/splits/manifest.json`
- Models/Configs: `configs/models/*.pkl`, `configs/scaler.pkl`, `configs/transform_metadata.json`
- Reports: `reports/baseline_metrics.csv`, `reports/baseline_coeffs.csv`
- Code: `src/models/splits.py`, `src/models/baseline.py`
- Tests: `tests/test_baseline.py`

## How to Run
- Regenerate features and splits:  
  `.venv/bin/python -m src.processing.pipeline`  
  `.venv/bin/python -m src.models.splits`
- Train baseline models:  
  `.venv/bin/python -m src.models.baseline`
- Tests:  
  `.venv/bin/pytest`

## Notes / Future
- Arrow emits `sysctlbyname ... Operation not permitted` warnings in this sandbox; harmless.
- Temperature/outdoor features are pruned by VIF; revisit if you want weather effects (e.g., different pruning or regularization focus).
