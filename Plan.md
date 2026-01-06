# ML Pricing Project Plan

## Purpose
- Build a reproducible pricing/elasticity workflow for café SKUs, replacing the ad-hoc notebook with scripts, clear data lineage, and collinearity-aware features.

## Scope
- In scope: data ingestion, EDA, feature engineering (log transforms, holiday/month encoding), correlation/VIF-based pruning, normalization, train/validation splits, baseline OLS + regularized model readiness.
- Out of scope (for now): deployment/serving, automated hyperparameter search, monitoring/alerting.

## Data Assets & Locations
- Raw sources: `Cafe+-+Transaction+-+Store.csv`, `Cafe+-+DateInfo.csv`, `Cafe+-+Sell+Meta+Data.csv` (copy to `data/raw/`; retain originals read-only).
- Intermediate/processed outputs: `data/processed/` (post-merge, encoded, normalized, split).
- Reports/figures: `reports/figures/` (EDA plots, correlation heatmaps, VIF tables).
- Code modules: `src/eda/`, `src/processing/`, `src/models/`, `src/utils/`; configs in `configs/`.

## Milestones and Tasks

## Sprint Completion Protocol
- After every successful sprint completion:
  - Update `Plan.md` checkboxes and note blockers or deviations.
  - Write a sprint summary at `docs/sprints/sprintN.md` covering what was done, why, and how.

### Sprint 1 – Data Understanding & EDA
- Goal: establish data health and correlation structure before feature pruning.
- Deliverables: EDA report/plots, correlation (Pearson/Spearman) summary, VIF baseline, data quality log.
- Tasks:
  - [x] Freeze raw CSVs into `data/raw/` with checksum or timestamp log.
  - [x] Data audit: missingness, duplicates, date ranges, category counts; log findings in `docs/` or `reports/`.
  - [x] Univariate + temporal EDA: distributions for `PRICE`, `QUANTITY`, `revenue`, per-SKU trends, holiday vs non-holiday averages; save plots under `reports/figures/`.
  - [x] Correlation analysis: Pearson and Spearman matrices; flag |ρ|>0.85 pairs; export heatmaps/tables.
  - [x] VIF baseline on encoded features (holiday/month dummies with reference drops); capture top offenders and notes.
  - [x] Testing: add sanity-check tests for audit utilities (e.g., duplicate-date detection, missing-value counters) under `tests/`.
- Definition of Done: EDA artifacts saved, issues noted (e.g., duplicate dates, skew), high-risk collinear pairs listed, `Plan.md` updated, sprint summary written.

### Sprint 2 – Data Transformation & Normalization
- Goal: build reusable transformation pipeline with collinearity pruning and scaling.
- Deliverables: ingestion + transform scripts, saved scaler/encoder config, pruned feature list with rationale.
- Tasks:
  - [x] Ingestion script (`src/processing/`): explicit date parsing (`%m/%d/%y`), deduplication of `CALENDAR_DATE`, merge transaction + date_info + product metadata.
  - [x] Feature engineering: fill holiday NAs, add `log_Q`, `log_p`, time parts (`MONTH`, `DAYOFWEEK`, `IS_MONTH_END`, `YEAR`), one-hot encode categorical fields with drop-first.
  - [x] Normalization: standardize continuous predictors (price log, temperature, interactions) with reusable scaler saved to `configs/` or `src/utils/`.
  - [x] Collinearity pruning routine: drop raw/log duplicates (keep logs), drop `YEAR` when month dummies present, collapse rare holidays, and iteratively remove features with VIF >10 then >5.
  - [x] Record transformation metadata (final feature list, scaler parameters, dropped-feature rationale) in `docs/` and/or `reports/`.
  - [x] Testing: unit tests for ingestion/transform steps (date parsing, deduplication, dummy encoding, scaler persistence) under `tests/`.
- Definition of Done: deterministic transformation script produces pruned, normalized dataset; metadata and configs saved.

### Sprint 3 – Modeling Readiness & Hygiene
- Goal: ship train/validation-ready datasets and baseline model scripts with tests.
- Deliverables: time-aware splits, baseline OLS/regularized scripts per SKU, tests for transformations.
- Tasks:
  - [x] Time-based splits per SKU (e.g., 80/20 by date) saved under `data/processed/` with manifest.
  - [x] Baseline model scripts (`src/models/`): OLS + optional Ridge/Lasso/Elastic Net; ensure each `SELL_ID` is modeled separately.
  - [x] Testing: unit tests for ingestion/transforms/scalers; sanity checks on feature columns and non-leakage; wire into `tests/`.
  - [x] Sprint close-out: update `Plan.md` and write sprint summary in `docs/sprints/`.
- Definition of Done: reproducible splits, runnable model scripts, passing tests, `Plan.md` updated, sprint summary written.

### Sprint 4 – Elasticity Deep Dive & Price Point Analysis
- Goal: stay aligned with log–log regression framing while producing decision-ready elasticity and revenue-maximizing price recommendations.
- Deliverables: elasticity tables per SKU, price-response curves, and candidate revenue-maximizing price points under observed ranges.
- Tasks:
  - [x] Extend baseline model scripts to compute/emit own-price elasticity from the log–log coefficients and confidence intervals.
  - [x] Generate price-response and revenue curves per SKU (using observed price range and model coefficients) and save plots/tables to `reports/`.
  - [x] Add robustness checks: re-fit log–log OLS/Ridge with/without select covariates (e.g., weather) to gauge elasticity stability.
  - [x] Document assumptions/limitations of the log–log approach (e.g., local elasticity, extrapolation bounds) in `docs/`.
  - [x] Tests: unit tests for elasticity extraction and revenue curve generation.
- Definition of Done: elasticity metrics and price-point recommendations saved to `reports/`, code/tests updated, summary in `docs/sprints/`.

### Sprint 5 – Validation, Reporting & What-Ifs
- Goal: validate model quality, quantify uncertainty, and package outputs for stakeholders.
- Deliverables: validation metrics beyond RMSE (e.g., R², MAPE), residual diagnostics, and a concise report on demand drivers and price sensitivity.
- Tasks:
  - [x] Add evaluation module to compute R², MAPE, and residual diagnostics per SKU; export to `reports/`.
  - [x] Perform time-based cross-validation or rolling-origin validation for the log–log models to assess temporal stability.
  - [x] Summarize key drivers and elasticity interpretations in a narrative report (`docs/`), keeping the model form consistent with the original description.
  - [x] Tests: add checks for metric calculations and that evaluation uses only train/val folds chronologically.
- Definition of Done: validation artifacts produced, report written, tests passing, sprint summary captured.

### Sprint 6 – Packaging & Handover (Optional)
- Goal: make the pipeline easy to rerun and hand off while preserving the log–log modeling approach.
- Deliverables: CLI/script entry points, refreshed documentation, and optional lightweight CI hooks.
- Tasks:
  - [x] Add simple CLI wrappers (e.g., `scripts/` or `python -m ...`) to run EDA, transforms, splits, and baseline models in sequence.
  - [x] Provide usage docs for rerunning the full pipeline and updating recommendations.
  - [x] (Optional) Wire pytest into a minimal CI workflow; ensure artifacts are documented/versioned.
  - [x] Tests: ensure CLI wrappers execute without errors in dry-run mode.
- Definition of Done: runnable end-to-end commands documented, optional CI stubbed, and handover notes prepared.

## Risks and Mitigations
- Multicollinearity: address via iterative VIF pruning and removal of redundant raw/log/year features.
- Sparse categories (rare holidays): mitigate by collapsing to “OtherHoliday” to stabilize dummies.
- Temporal leakage: avoid random splits; enforce chronological splits and validate feature availability at prediction time.

## Conventions
- All scripts run from repo root; outputs saved under `data/processed/` and `reports/figures/`.
- Use explicit seeds and write-protect raw data; prefer CSV/Parquet with schema notes in `docs/`.
- Environment: use a local virtual environment (`python -m venv .venv`) only; no global installs. Manage all dependencies via root-level `requirements.txt` (pin versions, update via PR/change log).
