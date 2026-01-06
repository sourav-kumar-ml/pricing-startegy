from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression, Ridge

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR, CONFIGS_DIR, ensure_dir


DATE_COLUMN = "CALENDAR_DATE"
TARGET_COLUMN = "log_Q"


def load_manifest(path: Path | None = None) -> List[dict]:
    path = path or (PROCESSED_DIR / "splits" / "manifest.json")
    data = json.loads(Path(path).read_text())
    return data


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {DATE_COLUMN, "SELL_ID", TARGET_COLUMN}
    return [c for c in df.columns if c not in drop_cols]


def train_and_eval(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, dict]:
    features = feature_columns(train_df)
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[features]
    y_val = val_df[TARGET_COLUMN]

    models = {
        "ols": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
    }

    metrics: Dict[str, dict] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_rmse = float(np.sqrt(((y_train - train_pred) ** 2).mean()))
        val_rmse = float(np.sqrt(((y_val - val_pred) ** 2).mean()))
        metrics[name] = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "coefficients": dict(zip(features, model.coef_)),
            "intercept": float(model.intercept_),
            "model": model,
        }
    return metrics


def save_results(
    all_metrics: Dict[int, Dict[str, dict]],
    metrics_path: Path | None = None,
    coefs_path: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    ensure_dir(REPORTS_DIR)
    ensure_dir(CONFIGS_DIR)
    metrics_path = metrics_path or (REPORTS_DIR / "baseline_metrics.csv")
    coefs_path = coefs_path or (REPORTS_DIR / "baseline_coeffs.csv")
    models_dir = models_dir or (CONFIGS_DIR / "models")
    ensure_dir(models_dir)

    rows = []
    coef_rows = []
    for sell_id, metric_map in all_metrics.items():
        for model_name, m in metric_map.items():
            rows.append(
                {
                    "SELL_ID": sell_id,
                    "model": model_name,
                    "train_rmse": m["train_rmse"],
                    "val_rmse": m["val_rmse"],
                    "intercept": m["intercept"],
                }
            )
            for feat, coef in m["coefficients"].items():
                coef_rows.append(
                    {
                        "SELL_ID": sell_id,
                        "model": model_name,
                        "feature": feat,
                        "coefficient": coef,
                    }
                )
            model_path = models_dir / f"{sell_id}_{model_name}.pkl"
            dump(m["model"], model_path)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    pd.DataFrame(coef_rows).to_csv(coefs_path, index=False)


def run_models(manifest_path: Path | None = None) -> None:
    manifest = load_manifest(manifest_path)
    all_metrics: Dict[int, Dict[str, dict]] = {}
    for entry in manifest:
        sell_id = entry["sell_id"]
        train_df = load_parquet(entry["train_path"])
        val_df = load_parquet(entry["val_path"])
        all_metrics[sell_id] = train_and_eval(train_df, val_df)
    save_results(all_metrics)


if __name__ == "__main__":
    run_models()
    print("Baseline models trained and metrics saved.")
