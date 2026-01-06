from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from src.utils.paths import CONFIGS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dir


DATE_COLUMN = "CALENDAR_DATE"
SELL_ID = "SELL_ID"
TARGET_COLUMN = "log_Q"


def load_manifest(path: Path | None = None) -> List[dict]:
    path = path or (PROCESSED_DIR / "splits" / "manifest.json")
    return json.loads(Path(path).read_text())


def load_metadata(path: Path | None = None) -> dict:
    path = path or (CONFIGS_DIR / "transform_metadata.json")
    return json.loads(Path(path).read_text())


def feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {DATE_COLUMN, SELL_ID, TARGET_COLUMN}
    return [c for c in df.columns if c not in drop_cols]


def evaluate_model(model, df: pd.DataFrame, split_label: str) -> Tuple[dict, pd.DataFrame]:
    feats = feature_columns(df)
    X = df[feats]
    y_true = df[TARGET_COLUMN]
    y_pred = model.predict(X)
    residuals = y_true - y_pred
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    # guard against zero targets for MAPE
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
    else:
        mape = np.nan
    metrics = {
        "split": split_label,
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape) if not np.isnan(mape) else None,
        "mean_residual": float(residuals.mean()),
    }
    resid_df = pd.DataFrame(
        {DATE_COLUMN: df[DATE_COLUMN], "y_true": y_true, "y_pred": y_pred, "residual": residuals, "split": split_label}
    )
    return metrics, resid_df


def rolling_origin_cv(df: pd.DataFrame, n_folds: int = 3) -> List[dict]:
    df_sorted = df.sort_values(DATE_COLUMN)
    feats = feature_columns(df_sorted)
    results: List[dict] = []
    n = len(df_sorted)
    fold_sizes = [n // (n_folds + 1)] * n_folds
    start = fold_sizes[0]
    for i in range(n_folds):
        cutoff = start + i * fold_sizes[0]
        if cutoff >= n:
            break
        train = df_sorted.iloc[:cutoff]
        val = df_sorted.iloc[cutoff : cutoff + fold_sizes[0]]
        if len(val) == 0 or len(train) < 2:
            continue
        model = LinearRegression()
        model.fit(train[feats], train[TARGET_COLUMN])
        y_pred = model.predict(val[feats])
        rmse = float(np.sqrt(mean_squared_error(val[TARGET_COLUMN], y_pred)))
        r2 = float(r2_score(val[TARGET_COLUMN], y_pred))
        results.append({"fold": i + 1, "rmse": rmse, "r2": r2, "val_rows": len(val)})
    return results


def run_evaluation() -> dict[str, Path]:
    manifest = load_manifest()
    metadata = load_metadata()
    ensure_dir(REPORTS_DIR)

    metrics_rows: List[dict] = []
    resid_rows: List[pd.DataFrame] = []
    cv_rows: List[dict] = []

    for entry in manifest:
        sid = int(entry["sell_id"])
        train_df = pd.read_parquet(entry["train_path"])
        val_df = pd.read_parquet(entry["val_path"])
        # pick up available models for this sell_id
        model_dir = CONFIGS_DIR / "models"
        model_paths = list(model_dir.glob(f"{sid}_*.pkl"))
        if not model_paths:
            continue
        for model_path in model_paths:
            model_name = model_path.stem.split("_", 1)[-1]
            model = load(model_path)
            for split_label, df in (("train", train_df), ("val", val_df)):
                metrics, resid_df = evaluate_model(model, df, split_label=split_label)
                metrics_rows.append({"SELL_ID": sid, "model": model_name, **metrics})
                resid_df["SELL_ID"] = sid
                resid_df["model"] = model_name
                resid_rows.append(resid_df)
        # rolling origin CV on the full SKU dataset (train+val concatenated)
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        for res in rolling_origin_cv(full_df):
            res.update({"SELL_ID": sid})
            cv_rows.append(res)

    metrics_df = pd.DataFrame(metrics_rows)
    resid_df_all = pd.concat(resid_rows, ignore_index=True) if resid_rows else pd.DataFrame()
    cv_df = pd.DataFrame(cv_rows)

    metrics_path = REPORTS_DIR / "eval_metrics.csv"
    residuals_path = REPORTS_DIR / "eval_residuals.csv"
    cv_path = REPORTS_DIR / "eval_rolling_cv.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not resid_df_all.empty:
        resid_df_all.to_csv(residuals_path, index=False)
    if not cv_df.empty:
        cv_df.to_csv(cv_path, index=False)

    return {"metrics": metrics_path, "residuals": residuals_path, "rolling_cv": cv_path}


if __name__ == "__main__":
    paths = run_evaluation()
    for name, path in paths.items():
        print(f"Wrote {name} to {path}")
