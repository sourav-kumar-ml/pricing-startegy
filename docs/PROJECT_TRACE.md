# Project Trace (End-to-End + File-by-File)

This is a “follow-the-wires” guide to understand exactly how the repo runs, how modules call each other, and how the pipeline passes state across steps.

## How to use this document (recommended order)
1. Start with the **entrypoint**: `scripts/pipeline.sh`.
2. Follow the pipeline stages in order:
   - `src/eda/run_eda.py`
   - `src/processing/pipeline.py`
   - `src/models/splits.py`
   - `src/models/baseline.py`
   - `src/models/elasticity.py`
   - `src/models/eval.py`
3. When you hit a helper call (like `read_transactions()` or `ensure_dir()`), jump to the corresponding `src/utils/*` module.
4. Use `docs/IMPORT_MAP.md` to see the code-level import graph (who imports whom).

## Mental model: “separation of concerns” in *this* repo

If you’re used to notebooks, you’re used to:
- One global, mutable runtime (variables hang around across cells).
- Linear execution in one place.
- In-memory objects passed implicitly (“I ran a cell earlier so the dataframe exists”).

This repo is structured differently on purpose:
- **Each “stage” is a module** you can run independently (`python -m ...`).
- **Stages communicate via artifacts on disk**, not via in-memory variables:
  - raw inputs in `data/raw/`
  - processed datasets in `data/processed/`
  - configs/models in `configs/`
  - reports/exports in `reports/`
- “Separation of concerns” here means:
  - `src/utils/`: reusable primitives (paths, IO, reporting)
  - `src/eda/`: exploratory + diagnostic outputs (audit, collinearity, plots)
  - `src/processing/`: building the model-ready feature matrix
  - `src/models/`: splitting, training, elasticity extraction, evaluation
  - `scripts/`: orchestration wrappers (bash, and optional helpers)

That design gives you reproducibility: you can delete `data/processed/` and `reports/` and rerun the same stages to regenerate everything.

## What “running a module” means (`python -m ...`)

When you run:
- `python -m src.processing.pipeline`

Python loads the module `src.processing.pipeline` and then executes the `if __name__ == "__main__": ...` block inside it.

This repo relies on “run from repo root” semantics:
- Running from the repo root puts the repo root on `sys.path`, so `import src...` works.
- Tests enforce this explicitly via `tests/conftest.py`.

## The pipeline at a glance (code + dataflow)

### Orchestrator
`scripts/pipeline.sh` runs stages in this strict order:
1. EDA: `python -m src.eda.run_eda`
2. Transform: `python -m src.processing.pipeline`
3. Splits: `python -m src.models.splits`
4. Baselines: `python -m src.models.baseline`
5. Elasticity/Curves: `python -m src.models.elasticity`
6. Evaluation: `python -m src.models.eval`

### Artifact/dataflow graph (the *real* coupling)
Most stages don’t import each other — they depend on each other via files:

```
data/raw/*.csv
  ├─(EDA)─ src/eda/*  ───────────────> reports/* (audit, correlations, VIF, figures)
  │
  └─(Transform)─ src/processing/pipeline.py
        ├───────────────────────────> data/processed/features.parquet
        ├───────────────────────────> data/processed/features_with_ids.parquet
        ├───────────────────────────> configs/scaler.pkl
        └───────────────────────────> configs/transform_metadata.json

data/processed/features_with_ids.parquet
  └─(Splits)─ src/models/splits.py
        ├───────────────────────────> data/processed/splits/<SELL_ID>_{train,val}.parquet
        └───────────────────────────> data/processed/splits/manifest.json

data/processed/splits/* + manifest.json
  └─(Baselines)─ src/models/baseline.py
        ├───────────────────────────> configs/models/<SELL_ID>_{ols,ridge}.pkl
        ├───────────────────────────> reports/baseline_metrics.csv
        └───────────────────────────> reports/baseline_coeffs.csv

configs/transform_metadata.json + data/processed/features_with_ids.parquet + data/raw/Cafe+-+DateInfo.csv
  └─(Elasticity)─ src/models/elasticity.py
        ├───────────────────────────> reports/elasticity.csv
        ├───────────────────────────> reports/price_curves.csv
        └───────────────────────────> reports/price_recommendations.csv

data/processed/splits/* + manifest.json + configs/models/*.pkl
  └─(Evaluation)─ src/models/eval.py
        ├───────────────────────────> reports/eval_metrics.csv
        ├───────────────────────────> reports/eval_residuals.csv
        └───────────────────────────> reports/eval_rolling_cv.csv
```

That artifact-level coupling is why the bash pipeline order matters.

## Core “data model” (what columns exist where)

### Raw inputs (`data/raw/`)
These are read via `src/utils/io.py`:
- `Cafe+-+DateInfo.csv` → `read_date_info()`
  - must include `CALENDAR_DATE` (parsed with `"%m/%d/%y"`)
  - used for `HOLIDAY`, weather/outdoor columns (e.g., `AVERAGE_TEMPERATURE`, `IS_OUTDOOR`)
- `Cafe+-+Transaction+-+Store.csv` → `read_transactions()`
  - must include `CALENDAR_DATE`, `SELL_ID`, `PRICE`, `QUANTITY`
  - includes additional flags like `IS_SCHOOLBREAK`, `IS_WEEKEND` (used/dropped later)
- `Cafe+-+Sell+Meta+Data.csv` → `read_product_meta()`
  - joins SKU metadata onto transactions by `SELL_ID`

### Processed dataset (`data/processed/features_with_ids.parquet`)
Produced by `src/processing/pipeline.py` and read by:
- `src/models/splits.py` (to create splits)
- `src/models/elasticity.py` (to estimate elasticity/curves)

It contains:
- meta columns: `CALENDAR_DATE`, `SELL_ID`
- target: `log_Q`
- features: one-hot encoded calendar features + `log_p` (scaled) + a few flags

### Split datasets (`data/processed/splits/*_{train,val}.parquet`)
Produced by `src/models/splits.py` and read by:
- `src/models/baseline.py` (to train models and write `configs/models/*.pkl`)
- `src/models/eval.py` (to evaluate trained models)

Each split parquet contains:
- meta columns: `CALENDAR_DATE`, `SELL_ID`
- target: `log_Q`
- features: same columns as in `features_with_ids.parquet` (minus any missing dummies for a specific SKU/date range)

## File-by-file walkthrough (what each file does, in detail)

### `scripts/pipeline.sh` (the end-to-end runner)
Role: *orchestration*.
- Ensures you’re in repo root, activates `.venv`, then runs each stage as a module.
- Sets `MPLBACKEND=Agg` for EDA so plots render headlessly (CI-friendly).

Key connection points:
- Calls the Python entrypoints; it does not import Python code.
- The order matters because later modules read files created by earlier modules.

---

### `src/utils/paths.py` (central directory layout)
Role: define canonical paths and provide a tiny directory helper.

Key exports:
- `REPO_ROOT`: computed from file location (`src/utils/paths.py` → repo root via `parents[2]`)
- `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `REPORTS_DIR`, `FIGURES_DIR`, `CONFIGS_DIR`
- `ensure_dir(path: Path)`: creates directories (parents ok), used everywhere before writing outputs

Connections:
- Imported by almost every stage so paths aren’t hardcoded.

---

### `src/utils/io.py` (raw CSV readers)
Role: one place for raw-data parsing rules.

Key constants:
- `DATE_FORMAT = "%m/%d/%y"`
- `DATE_COLUMN = "CALENDAR_DATE"`
- raw filenames: `RAW_DATE_INFO`, `RAW_TRANSACTIONS`, `RAW_PRODUCT_META`

Key functions:
- `_resolve(path)`: if given a string, resolves relative to `RAW_DIR`; if `Path`, uses it directly.
- `read_date_info()`: `pd.read_csv(...)` + parses `CALENDAR_DATE` with `DATE_FORMAT`.
- `read_transactions()`: same date parsing.
- `read_product_meta()`: plain csv read (no date parsing).

Connections:
- Used by EDA (`src/eda/audit.py`, `src/eda/collinearity.py`), processing (`src/processing/pipeline.py`), and elasticity robustness (`src/models/elasticity.py`).

---

### `src/utils/reporting.py` (markdown writer)
Role: tiny formatting helper for “report-like” outputs.

Key function:
- `write_markdown(path, title, sections)`: writes a Markdown doc with `# Title` and `## Section` headers.

Connections:
- Used only by `src/eda/audit.py` to write `reports/eda_audit.md`.

---

### `src/eda/audit.py` (EDA: dataset audit report)
Role: produce a human-readable audit summary.

Main inputs:
- `read_date_info()`, `read_transactions()`, `read_product_meta()` from `src/utils/io.py`

Core logic:
- `summarize_dataframe(df, name, date_column=None)`:
  - counts rows, columns, duplicate rows, missing cells
  - if `date_column` present: min/max date and number of unique dates
- `build_audit_sections(...)`:
  - builds Markdown sections:
    - dataset overview (size, missingness, duplicates, date range)
    - holiday coverage (missing/NULL + top values)
    - top `SELL_ID` counts
    - `SELL_CATEGORY` counts (if present)
- `run_audit(output_path=None)`:
  - reads raw datasets
  - writes `reports/eda_audit.md`

Outputs:
- `reports/eda_audit.md` (written via `src/utils/reporting.write_markdown`)

How it’s called:
- `src/eda/run_eda.py` calls `run_audit()` inside `run_all()`.

---

### `src/eda/collinearity.py` (EDA: correlations + VIF)
Role: quantify collinearity risk before modeling.

Core data construction:
- `build_master()`:
  - reads date_info + transactions
  - merges on `CALENDAR_DATE`
  - normalizes `HOLIDAY`: converts `"NULL"` to NA, then fills NA with `"Not Holiday"`
  - adds `MONTH`
  - adds `log_Q`, `log_p` using `np.where` to guard non-positive values

Correlation outputs:
- `correlation_matrix(df, method)`: wrapper around `df.corr(method=...)`
- `top_correlated_pairs(corr, threshold=0.85, top_n=50)`:
  - iterates upper triangle of correlation matrix
  - emits pairs where `abs(corr) >= threshold`

VIF outputs:
- `prepare_vif_features(master)`:
  - one-hot encodes `HOLIDAY` and `MONTH`
  - keeps numeric columns only
  - drops columns that would contaminate VIF or are identifiers/targets (`SELL_ID`, `SELL_CATEGORY`, `QUANTITY`, `log_Q`)
  - drops zero-variance columns
- `compute_vif(feature_df)`:
  - calls `statsmodels.stats.outliers_influence.variance_inflation_factor` per column

Entrypoint:
- `run_collinearity(output_dir=None)`:
  - builds master
  - writes:
    - `reports/pearson_corr.csv`
    - `reports/spearman_corr.csv`
    - `reports/top_correlations.csv`
    - `reports/vif.csv`

How it’s called:
- `src/eda/run_eda.py` calls `run_collinearity()` inside `run_all()`.

---

### `src/eda/run_eda.py` (EDA: one command to run everything)
Role: convenience stage executed by the bash pipeline.

Imports:
- `run_audit` from `src.eda.audit`
- `build_master`, `run_collinearity` from `src.eda.collinearity`
- `FIGURES_DIR`, `ensure_dir` from `src.utils.paths`

Core entrypoint:
- `run_all()` does three things:
  1. `run_audit()` → generates `reports/eda_audit.md`
  2. `run_collinearity()` → generates correlation/VIF CSVs
  3. `generate_plots(master, FIGURES_DIR)`:
     - builds `master` again, adds `revenue`, writes plots under `reports/figures/`

Plotting details (`generate_plots`):
- Imports matplotlib/seaborn lazily (so base pipeline can run without plotting deps in some contexts).
- Produces:
  - price, quantity, revenue histograms
  - daily revenue lines per SKU
  - holiday vs non-holiday mean revenue bar plot

Outputs:
- report and csvs listed above + PNGs under `reports/figures/`.

---

### `src/processing/pipeline.py` (the transformation/feature pipeline)
Role: convert raw transaction logs into a model-ready, per-row feature matrix.

This is the “heart” of the repo: everything downstream assumes this stage has been run.

#### Key conventions
- Target column: `log_Q = log(QUANTITY)`
- Price feature: `log_p = log(PRICE)`
- Categorical encoding:
  - `HOLIDAY`, `MONTH`, `DAYOFWEEK` → one-hot dummies (`drop_first=True`)
- Collinearity handling:
  - iterative VIF pruning (drop features above thresholds, protect `log_p`)
- Scaling:
  - `StandardScaler` applied to float predictors that remain after pruning
  - scaling metadata is saved to `configs/transform_metadata.json`

#### Call tree when run as a module
When you execute `python -m src.processing.pipeline`:
1. `run_pipeline(...)`
2. `build_feature_matrix(...)`
3. `save_outputs(...)`

#### Detailed function-by-function behavior
- `load_raw_data()`:
  - reads three raw tables via `src.utils.io`
  - returns `(date_info, transactions, product)`

- `merge_raw(date_info, transactions, product)`:
  - de-dupes `date_info` on `CALENDAR_DATE`
  - converts holiday string `"NULL"` to NA
  - de-dupes `product` on `SELL_ID`
  - merges:
    - `transactions ⟕ date_info` on date
    - result `⟕ product` on `SELL_ID`

- `engineer_features(df, rare_holiday_min_count=20)`:
  - ensures a clean `HOLIDAY`:
    - fills missing with `"Not Holiday"`
    - collapses rare holiday categories into `"OtherHoliday"` (via `collapse_rare_categories`)
  - removes rows with non-positive `PRICE` or `QUANTITY` (logs require positivity)
  - adds:
    - `revenue = PRICE * QUANTITY`
    - `log_Q = log(QUANTITY)`
    - `log_p = log(PRICE)`
    - calendar features: `MONTH`, `DAYOFWEEK`, `IS_MONTH_END`, `YEAR`
    - interaction: `TEMP_X_OUTDOOR = AVERAGE_TEMPERATURE * IS_OUTDOOR`

- `encode_features(df)`:
  - `pd.get_dummies(..., columns=["HOLIDAY","MONTH","DAYOFWEEK"], drop_first=True, dtype=int)`
  - produces dummy columns like `HOLIDAY_New Year`, `MONTH_12`, `DAYOFWEEK_6`

- `select_feature_columns(df)`:
  - drops columns that are not features or are leakage/identifiers:
    - raw: `PRICE`, `QUANTITY`, `revenue`
    - identifiers/meta: `CALENDAR_DATE`, `SELL_ID`, `ITEM_*`, `SELL_CATEGORY*`
    - redundant: `YEAR`, `IS_WEEKEND`
  - removes the explicit “Not Holiday” dummy if present (`HOLIDAY_Not Holiday`)
  - pops and returns the target `log_Q` separately

- `iterative_vif_prune(df, thresholds=(10,5), protect={...})`:
  - for each threshold:
    - computes VIF for current set of columns
    - drops the highest-VIF column above threshold that is not in `protect`
    - repeats until all VIFs are below threshold or only 1 column remains
  - returns `(pruned_df, drops)` where `drops` is a structured record of what was dropped and why

- `scale_features(df, scale_cols)`:
  - fits `StandardScaler` on `scale_cols` that are present
  - writes scaled values back into the same dataframe columns
  - returns `(df, scaler_or_none)`

- `build_feature_matrix(...)`:
  - orchestrates all prior steps
  - additionally:
    - extracts “meta columns” (`CALENDAR_DATE`, `SELL_ID`) *before* dropping them from features
    - drops zero-variance features before VIF (important: VIF breaks on constants)
  - returns:
    - `dataset`: `[log_Q] + features` (no meta columns)
    - `dataset_with_meta`: `[CALENDAR_DATE, SELL_ID] + dataset`
    - `TransformationMetadata`: full lineage (what dropped, final features, scaling, etc.)
    - `scaler`: fitted scaler or `None`

- `save_outputs(...)`:
  - ensures directories exist
  - writes:
    - `data/processed/features.parquet`
    - `data/processed/features_with_ids.parquet`
    - `configs/scaler.pkl` (if any scaling happened)
    - `configs/transform_metadata.json`

#### Why metadata exists
`configs/transform_metadata.json` is the “contract” between processing and modeling:
- it records the final feature list and which features were scaled
- elasticity estimation uses it to:
  - choose feature sets consistently
  - back-transform the `log_p` coefficient into a real elasticity if `log_p` was standardized

---

### `src/models/splits.py` (time-aware train/val splitting)
Role: create per-SKU, chronological splits (no random leakage).

Core logic:
- `load_features_with_ids()` reads `data/processed/features_with_ids.parquet` and enforces that it contains:
  - `CALENDAR_DATE`
  - `SELL_ID`
- `time_based_split(df, ratio=0.8)`:
  - sorts by `CALENDAR_DATE`
  - splits by index at `int(len * ratio)` with guards to avoid empty train/val
- `split_by_sku(df, ratio=0.8)`:
  - loops `for sell_id, group in df.groupby("SELL_ID")`
  - applies `time_based_split`
  - writes:
    - `data/processed/splits/<SELL_ID>_train.parquet`
    - `data/processed/splits/<SELL_ID>_val.parquet`
  - builds a `SplitManifestEntry` containing paths + row counts + date ranges
- `save_manifest(...)` writes `data/processed/splits/manifest.json` as a list of entries.

How it’s called:
- `python -m src.models.splits` runs `run_splits()` which calls the above.

Downstream dependencies:
- Baseline training and evaluation use `manifest.json` to find split files.

---

### `src/models/baseline.py` (baseline regression training)
Role: train simple per-SKU models (OLS + Ridge) and persist them.

Inputs:
- `data/processed/splits/manifest.json` (paths to per-SKU split parquets)
- per-SKU split parquet files

Core logic:
- `feature_columns(df)`:
  - drops meta + target: `CALENDAR_DATE`, `SELL_ID`, `log_Q`
  - returns all remaining columns as features
- `train_and_eval(train_df, val_df)`:
  - trains two models:
    - `LinearRegression()` (labeled `"ols"`)
    - `Ridge(alpha=1.0)` (labeled `"ridge"`)
  - computes RMSE on train and val
  - stores coefficients and intercept
  - returns a dict that also contains the fitted model objects
- `save_results(...)`:
  - writes model pickles:
    - `configs/models/<SELL_ID>_ols.pkl`
    - `configs/models/<SELL_ID>_ridge.pkl`
  - writes reports:
    - `reports/baseline_metrics.csv`
    - `reports/baseline_coeffs.csv`

How it’s called:
- `python -m src.models.baseline` runs `run_models()`.

Downstream dependency:
- `src/models/eval.py` loads these `.pkl` files to evaluate on train/val and export residuals.

---

### `src/models/elasticity.py` (elasticity extraction + price/revenue curves)
Role: estimate own-price elasticity and generate decision-oriented price curves.

Important: this module **fits its own OLS models (statsmodels)** per SKU to extract coefficients + confidence intervals.
It does **not** reuse the sklearn models produced by `src/models/baseline.py`.

Inputs:
- `configs/transform_metadata.json`:
  - `final_features` list
  - `scaled_features` list (to determine if `log_p` was standardized)
  - `scaler_path` for back-transforming elasticity
  - `features_with_meta_path` pointing to `data/processed/features_with_ids.parquet`
- `data/processed/features_with_ids.parquet`:
  - provides per-row `log_Q`, `log_p`, and dummy features
- `data/raw/Cafe+-+DateInfo.csv`:
  - merged in again for optional weather robustness features (`AVERAGE_TEMPERATURE`, `IS_OUTDOOR`, `TEMP_X_OUTDOOR`)

Key internal pieces:
- `LogPScaling`:
  - encapsulates scaler mean/scale for `log_p`
  - provides `scale_log_p()` and `unscale_log_p()` helpers

- `load_log_p_scaling(metadata)`:
  - if `log_p` is in `metadata["scaled_features"]`, loads `configs/scaler.pkl`
  - extracts the mean/scale for the `log_p` feature
  - this enables correct elasticity back-transform:
    - if `log_p_scaled = (log_p - mean) / scale`
    - and the fitted coefficient is `β_scaled`,
    - then elasticity w.r.t. the true `log_p` is `β_scaled / scale`

- `get_feature_sets(metadata, df)`:
  - constructs multiple “models” to run per SKU:
    - `"full"`: uses `final_features` from metadata (ensures `log_p` is included)
    - `"price_only"`: uses only `log_p`
    - `"price_weather"`: uses `log_p` + weather features if they exist

- `fit_ols(df, features)`:
  - `statsmodels.api.OLS` with an explicit constant
  - returns `(results, used_feature_names)`

- `elasticity_from_results(...)`:
  - pulls the `log_p` coefficient from `results.params`
  - reads the confidence interval for `log_p` from `results.conf_int()`
  - back-transforms if `log_p` was scaled (divide by scale)

Price curve logic:
- `price_grid_from_logp(df, num=25, log_p_scaling=...)`:
  - takes the observed `log_p` column for the SKU
  - if scaled, unscales back to true `log_p`
  - converts to prices via `exp(log_p)`
  - builds a grid from min to max observed price (bounded; no extrapolation beyond observed range)

- `predict_curve(results, features, base_row, prices, ...)`:
  - builds a design matrix over the price grid
  - uses a “base row” (median values of other features) to hold non-price features constant
  - predicts `log_Q`, exponentiates to get `Q̂`, computes `revenue = price * Q̂`

Outputs (written under `reports/`):
- `elasticity.csv` (one row per SKU × model type)
- `price_curves.csv` (rows for each SKU × model × price grid point)
- `price_recommendations.csv` (argmax revenue per SKU × model)

How it’s called:
- `python -m src.models.elasticity` runs `run_elasticity()`.

---

### `src/models/eval.py` (evaluate saved baseline models + diagnostics)
Role: evaluate the persisted sklearn models from `configs/models/` on train/val splits.

Inputs:
- `data/processed/splits/manifest.json` (locates the split parquet files)
- `configs/models/<SELL_ID>_*.pkl` (trained models from `src/models/baseline.py`)

Core evaluation functions:
- `feature_columns(df)`:
  - same logic as baseline: drop `CALENDAR_DATE`, `SELL_ID`, `log_Q`
- `evaluate_model(model, df, split_label)`:
  - predicts `y_pred = model.predict(X)`
  - computes:
    - RMSE
    - R²
    - MAPE (guarded for `y_true != 0`)
    - mean residual
  - returns:
    - `metrics` dict
    - `resid_df` with date + y_true/y_pred/residual/split

Rolling-origin CV:
- `rolling_origin_cv(df, n_folds=3)`:
  - sorts by date
  - creates sequential train/val blocks
  - fits fresh `LinearRegression()` models (not the persisted ones) on each fold
  - returns fold-level RMSE/R²

Entrypoint `run_evaluation()`:
- For each SKU in the manifest:
  - loads train + val parquet
  - finds model files in `configs/models/` matching `<SELL_ID>_*.pkl`
  - evaluates each model on train and val
  - runs rolling-origin CV on `train ∪ val`
- Writes:
  - `reports/eval_metrics.csv`
  - `reports/eval_residuals.csv` (if any residuals were produced)
  - `reports/eval_rolling_cv.csv`

How it’s called:
- `python -m src.models.eval` runs `run_evaluation()`.

---

## Tests: what each one proves (and which module it targets)

### `tests/conftest.py`
Role: ensures `import src...` works by inserting the repo root into `sys.path`.

### `tests/test_pipeline.py` → `src/processing/pipeline.py`
Validates:
- rare category collapsing
- log feature engineering + encoding + column selection
- VIF pruning drops at least one collinear feature in a toy case
- scaling produces ~zero mean and unit variance

### `tests/test_audit.py` → `src/eda/audit.py`
Validates:
- dataframe summary counts (rows/cols/duplicates/missing/date range)
- `format_counts` formatting

### `tests/test_collinearity.py` → `src/eda/collinearity.py`
Validates:
- VIF feature preparation drops target/IDs correctly

### `tests/test_splits.py` → `src/models/splits.py`
Validates:
- chronological ordering and split sizing of `time_based_split`

### `tests/test_baseline.py` → `src/models/baseline.py`
Validates:
- feature selection excludes meta/target
- training/eval runs end-to-end and returns coefficients

### `tests/test_elasticity.py` → `src/models/elasticity.py`
Validates:
- OLS fitting + elasticity extraction
- curve prediction output shapes
- feature set selection includes weather when present
- correct back-transform when `log_p` is standardized
- correct price-grid behavior when `log_p` is standardized

### `tests/test_eval.py` → `src/models/eval.py`
Validates:
- feature column filtering
- evaluation metrics and residual outputs
- rolling-origin CV produces at least one fold

## Where to look if you want to change behavior

Common modifications map cleanly to one module:
- Add/remove features: `src/processing/pipeline.py` (`engineer_features`, `encode_features`, `select_feature_columns`)
- Change collinearity strategy: `src/processing/pipeline.py` (`iterative_vif_prune`, VIF thresholds)
- Change split logic: `src/models/splits.py` (`time_based_split`)
- Add a new model: `src/models/baseline.py` (add to `models` dict) and/or `src/models/elasticity.py` (add feature set)
- Change evaluation metrics: `src/models/eval.py` (`evaluate_model`)

