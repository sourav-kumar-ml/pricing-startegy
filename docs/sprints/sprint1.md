# Sprint 1 Summary - Data Understanding and EDA (Status: Completed)

## Scope and Intent
- Establish canonical raw data storage and a scriptable EDA workflow.
- Provide repeatable audit and collinearity diagnostics without relying on notebooks.
- Lay the foundation for later feature pruning and normalization.

## Work Completed

### 1) Raw data staging
- What: moved all source CSVs into `data/raw/` and created a checksum log.
- Why: enforce a single, read-only source of truth for downstream scripts.
- How: relocated the files and generated SHA-256 checksums in `data/raw/CHECKSUMS.txt`.

### 2) Data loading utilities
- What: added shared loaders and path helpers for consistent reads.
- Why: avoid inconsistent date parsing and hard-coded paths across scripts.
- How: implemented `src/utils/paths.py` for directory constants and `src/utils/io.py`
  for explicit `%m/%d/%y` parsing of `CALENDAR_DATE`.

### 3) Audit module
- What: built a structured audit script that summarizes row counts, duplicates,
  missing cells, and date ranges.
- Why: capture dataset health in a persistent report instead of ad-hoc checks.
- How: implemented `src/eda/audit.py` with a `DataframeAudit` dataclass and a
  markdown report writer (`reports/eda_audit.md`).

### 4) Collinearity diagnostics
- What: created correlation and VIF outputs for linear and monotonic relationships.
- Why: provide evidence-driven pruning targets (Pearson, Spearman, VIF).
- How: `src/eda/collinearity.py` builds a merged master dataset, applies log
  transforms for `PRICE`/`QUANTITY`, encodes `HOLIDAY`/`MONTH`, and exports
  correlation matrices plus VIF tables to `reports/`.

### 5) EDA runner and plots
- What: a single entry point for audit + collinearity + standard plots.
- Why: reduce manual steps and enable repeatable EDA runs.
- How: `src/eda/run_eda.py` orchestrates the audit/collinearity scripts and
  generates histograms and time-series plots under `reports/figures/`.
- Result: ran the pipeline (with `MPLBACKEND=Agg`) producing:
  - `reports/eda_audit.md`
  - `reports/pearson_corr.csv`, `reports/spearman_corr.csv`, `reports/top_correlations.csv`
  - `reports/vif.csv`
  - Plots in `reports/figures/` (price/quantity/revenue distributions, daily revenue by SKU, holiday vs non-holiday revenue)

### 6) Testing baseline
- What: added initial unit tests for EDA utilities.
- Why: protect key audit/feature prep logic from regressions.
- How: tests in `tests/test_audit.py` and `tests/test_collinearity.py`.

### 7) Dependency management
- What: created a root `requirements.txt`.
- Why: enforce venv-only installs with a reproducible dependency list.
- How: pinned minimum versions for core EDA libraries and pytest.

## Artifacts Produced
- `data/raw/CHECKSUMS.txt`
- `src/utils/paths.py`
- `src/utils/io.py`
- `src/utils/reporting.py`
- `src/eda/audit.py`
- `src/eda/collinearity.py`
- `src/eda/run_eda.py`
- `reports/eda_audit.md`
- `reports/pearson_corr.csv`
- `reports/spearman_corr.csv`
- `reports/top_correlations.csv`
- `reports/vif.csv`
- `reports/figures/*.png`
- `tests/test_audit.py`
- `tests/test_collinearity.py`
- `requirements.txt`

## How to Run
- EDA pipeline: `MPLBACKEND=Agg .venv/bin/python -m src.eda.run_eda`
- Tests: `pytest`

## Remaining Sprint 1 Tasks
- None; sprint closed. Minor note: matplotlib warned about `~/.matplotlib` not writable; consider setting `MPLCONFIGDIR` to a writable cache dir for faster imports.
