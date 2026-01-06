# Sprint 5 Summary - Validation, Reporting & What-Ifs (Status: Completed)

## Scope and Intent
- Validate log–log baseline models with richer metrics, residual diagnostics, and rolling temporal checks.
- Produce evaluation artifacts for stakeholders while keeping the original modeling form.

## Work Completed

### 1) Evaluation module
- What: Added `src/models/eval.py` to load saved models and splits, compute RMSE/R²/MAPE and residual stats for train/val, and run rolling-origin checks.
- Outputs: `reports/eval_metrics.csv`, `reports/eval_residuals.csv`, `reports/eval_rolling_cv.csv`.

### 2) Metrics and residuals
- What: Stored per-SKU, per-model metrics and residuals (with dates) to support diagnostics and plotting.
- Why: Make quality and error patterns inspectable without notebooks.

### 3) Rolling-origin validation
- What: Simple forward-in-time folds to gauge temporal stability beyond a single split.
- Why: Reduce leakage risk and surface time drift in elasticities.

### 4) Testing
- What: Added tests for evaluation utilities.
- How: `tests/test_eval.py`; full suite passes.

## How to Run
- Evaluation: `.venv/bin/python -m src.models.eval`
- Tests: `.venv/bin/pytest`

## Notes / Future
- Rolling folds are lightweight; increase fold count or adjust window size if needed.
- MAPE skips zero targets; consider alternative scaled errors if zeros become common.
