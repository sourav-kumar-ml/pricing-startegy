# Pricing ML Project

A scripted, notebook-free workflow for estimating SKU-level price elasticity from historical café transactions and producing decision-oriented outputs (elasticities, price/revenue curves, and price recommendations) with reproducible data lineage.

## Project Description (What This Does)
This repository turns raw daily transactions into **per-SKU demand models** and **pricing sensitivity diagnostics**:
- Ingests and merges raw transaction, calendar, and product metadata CSVs.
- Builds a feature matrix with log transforms and seasonal/event covariates.
- Applies collinearity diagnostics (correlations + VIF) and VIF-based feature pruning.
- Creates **time-aware** train/validation splits per SKU (no random shuffling).
- Fits simple baseline regressions (OLS and Ridge) on `log_Q` (log quantity).
- Estimates own-price elasticity from log–log regressions and generates **price/quantity/revenue curves** over the **observed** price range.

If you want the narrative summary (interpretation + limitations), start with `docs/report.md` and `docs/assumptions.md`.

## Goal & Scope
**Goal:** quantify how demand responds to price for each SKU (`SELL_ID`) and generate actionable, bounded pricing guidance that is easy to reproduce and audit.

**In scope**
- EDA and data-quality auditing (missingness, duplicates, date coverage).
- Feature engineering for price/demand modeling (log transforms, seasonality, holidays, simple interactions).
- Multicollinearity handling via correlations and VIF-based pruning.
- Per-SKU time-aware splits, baseline model training, elasticity estimation, and evaluation diagnostics.

**Out of scope (by design / not implemented here)**
- Causal pricing inference (e.g., randomized experiments, IV methods) and any claim that coefficients are causal effects.
- Production deployment/serving, monitoring, and automated retraining.
- Hyperparameter search, advanced ML models, or pooled/hierarchical cross-SKU models.

## Statistical Methods Implemented
The core model family is a **log–log demand regression** fit separately per SKU:

`log(Q_t) = β0 + βp·log(P_t) + βᵀX_t + ε_t`  → **own-price elasticity** ≈ `βp`

Implemented methods and diagnostics:
- **Log transforms:** `log_Q = log(QUANTITY)` and `log_p = log(PRICE)` (positive values only).
- **Categorical encoding:** one-hot dummies for `HOLIDAY`, `MONTH`, `DAYOFWEEK` with drop-first encoding.
- **Correlation analysis:** Pearson + Spearman correlation matrices and top correlated pairs (`reports/pearson_corr.csv`, `reports/spearman_corr.csv`, `reports/top_correlations.csv`).
- **VIF diagnostics and pruning:** variance inflation factor (VIF) computed on encoded features; iterative drops above thresholds (default 10 then 5) while protecting `log_p`.
- **Normalization:** `StandardScaler` applied to continuous predictors selected by the pipeline (see `configs/transform_metadata.json`); elasticity/curve generation back-transforms so outputs remain in real price units.
- **Time-aware validation:** chronological 80/20 split per SKU + rolling-origin CV for stability checks.
- **Models:** OLS (ordinary least squares) and Ridge regression baselines.
- **Uncertainty:** elasticity confidence intervals from `statsmodels` OLS.
- **Price curves:** simulated price grid over each SKU’s observed price range; predicts `Q` via `exp(predicted log_Q)` and computes modeled revenue `price * Q̂`.
- **Evaluation:** RMSE, R², MAPE, and residual exports on train/validation splits (`reports/eval_metrics.csv`, `reports/eval_residuals.csv`, `reports/eval_rolling_cv.csv`).

## Repo Layout
```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── models/                      # <SELL_ID>_{ols,ridge}.pkl
│   ├── scaler.pkl                   # StandardScaler saved by preprocessing
│   └── transform_metadata.json      # features, VIF drops, artifact paths
├── data/
│   ├── raw/                         # immutable input CSVs + checksums
│   │   ├── Cafe+-+DateInfo.csv
│   │   ├── Cafe+-+Sell+Meta+Data.csv
│   │   ├── Cafe+-+Transaction+-+Store.csv
│   │   └── CHECKSUMS.txt
│   └── processed/
│       ├── features.parquet
│       ├── features_with_ids.parquet
│       └── splits/
│           └── manifest.json
├── docs/
│   ├── assumptions.md
│   ├── report.md
│   └── sprints/
├── reports/
│   ├── figures/                     # plots from EDA
│   ├── eda_audit.md
│   ├── pearson_corr.csv
│   ├── spearman_corr.csv
│   ├── top_correlations.csv
│   ├── vif.csv
│   ├── baseline_metrics.csv
│   ├── baseline_coeffs.csv
│   ├── elasticity.csv
│   ├── price_curves.csv
│   ├── price_recommendations.csv
│   ├── eval_metrics.csv
│   ├── eval_residuals.csv
│   └── eval_rolling_cv.csv
├── scripts/
│   └── pipeline.sh                  # end-to-end runner
├── src/
│   ├── eda/                         # audit + collinearity + plots
│   ├── models/                      # splits, baselines, elasticity, evaluation
│   ├── processing/                  # feature pipeline
│   └── utils/
├── tests/
├── Plan.md
├── README.md
└── requirements.txt
```

## Setup
1) Python 3.11+ (CI runs 3.11); create/activate venv: `python -m venv .venv && source .venv/bin/activate`
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
