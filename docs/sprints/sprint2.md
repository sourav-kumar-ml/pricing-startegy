# Sprint 2 Summary - Data Transformation & Normalization (Status: Completed)

## Scope and Intent
- Build a reusable, scriptable transformation pipeline with feature engineering, collinearity pruning, and normalization.
- Persist processed datasets and transformation metadata/configs for downstream modeling.

## Work Completed

### 1) Ingestion and merge
- What: Deterministic ingestion that parses dates (`%m/%d/%y`), deduplicates calendar rows, and merges transactions with date info and product metadata (deduped on `SELL_ID`).
- Why: Prevent row explosion and enforce consistent date handling for all downstream steps.
- How: `src/processing/pipeline.py` (`merge_raw`, `load_raw_data`) with explicit deduplication on `CALENDAR_DATE` and `SELL_ID`.

### 2) Feature engineering
- What: Added log transforms (`log_Q`, `log_p`), revenue, calendar parts (`MONTH`, `DAYOFWEEK`, `IS_MONTH_END`, `YEAR`), outdoor interaction (`TEMP_X_OUTDOOR`), and holiday cleanup (fill NAs, collapse rare holidays to `OtherHoliday`).
- Why: Provide elasticities-friendly targets and seasonality/holiday signals while handling sparse categories.
- How: `engineer_features` in `src/processing/pipeline.py`.

### 3) Encoding and collinearity pruning
- What: One-hot encoding for `HOLIDAY`, `MONTH`, `DAYOFWEEK` (drop-first), manual baseline drops (`HOLIDAY_Not Holiday`, duplicate `SELL_CATEGORY` cols, `IS_WEEKEND`), then iterative VIF pruning (thresholds 10 then 5) while protecting `log_p`.
- Why: Reduce multicollinearity and keep interpretable price elasticity features.
- How: `encode_features`, `select_feature_columns`, and `iterative_vif_prune` in `src/processing/pipeline.py`.

### 4) Normalization and outputs
- What: Standardized continuous feature `log_p` (kept as a protected feature), saved scaler and metadata, and wrote processed features dataset.
- Why: Ensure downstream models can reuse the same scaling and feature set reproducibly.
- How: `scale_features`, `save_outputs` in `src/processing/pipeline.py`; outputs: `data/processed/features.parquet`, `configs/scaler.pkl`, `configs/transform_metadata.json`.

### 5) Testing
- What: Added unit tests for collapse logic, feature engineering, VIF pruning, and scaling.
- Why: Guard against regressions in the transformation pipeline.
- How: `tests/test_pipeline.py` plus existing EDA tests; suite passes with `.venv/bin/pytest`.

### 6) Dependencies
- What: Added `scikit-learn` (for scaling) and `pyarrow` (for Parquet) to `requirements.txt`.
- Why: Support normalization and efficient dataset persistence.

## Artifacts Produced
- `src/processing/pipeline.py` (ingestion, feature engineering, encoding, VIF pruning, scaling, save)
- `data/processed/features.parquet`
- `configs/scaler.pkl`
- `configs/transform_metadata.json`
- Updated `requirements.txt`
- Tests: `tests/test_pipeline.py` (plus prior EDA tests)

## How to Run
- Transformation pipeline: `.venv/bin/python -m src.processing.pipeline`
- Tests: `.venv/bin/pytest`

## Notes / Next Considerations
- VIF pruning dropped temperature-based features (`AVERAGE_TEMPERATURE`, `TEMP_X_OUTDOOR`, `IS_OUTDOOR`) due to high multicollinearity; revisit if you want to keep a weather signal (e.g., by removing other correlated columns or regularizing instead of dropping).
- Matplotlib cache warning from Sprint 1 can be silenced by setting `MPLCONFIGDIR` to a writable folder if needed.

## Remaining Sprint 2 Tasks
- None; sprint closed.
