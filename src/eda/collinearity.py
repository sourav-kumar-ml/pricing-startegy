from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.io import read_date_info, read_transactions
from src.utils.paths import REPORTS_DIR, ensure_dir


DATE_COLUMN = "CALENDAR_DATE"


def build_master() -> pd.DataFrame:
    date_info = read_date_info().drop_duplicates(subset=[DATE_COLUMN])
    transactions = read_transactions()
    master = transactions.merge(date_info, on=DATE_COLUMN, how="left")
    master["HOLIDAY"] = master["HOLIDAY"].replace("NULL", pd.NA).fillna("Not Holiday")
    master["MONTH"] = master[DATE_COLUMN].dt.month

    master["log_Q"] = np.where(master["QUANTITY"] > 0, np.log(master["QUANTITY"]), np.nan)
    master["log_p"] = np.where(master["PRICE"] > 0, np.log(master["PRICE"]), np.nan)
    return master


def correlation_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df.corr(method=method)


def top_correlated_pairs(
    corr: pd.DataFrame, threshold: float = 0.85, top_n: int = 50
) -> pd.DataFrame:
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                pairs.append((cols[i], cols[j], float(value)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    pairs = pairs[:top_n]
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"])


def prepare_vif_features(master: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(master, columns=["HOLIDAY", "MONTH"], drop_first=True, dtype=int)
    numeric = encoded.select_dtypes(include="number")
    drop_cols = {"SELL_ID", "SELL_CATEGORY", "QUANTITY", "log_Q"}
    numeric = numeric.drop(columns=[c for c in drop_cols if c in numeric.columns])
    zero_var = [c for c in numeric.columns if numeric[c].nunique() <= 1]
    return numeric.drop(columns=zero_var)


def compute_vif(feature_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError as exc:
        raise RuntimeError("statsmodels is required to compute VIFs") from exc

    values = feature_df.values
    vifs = []
    for i, column in enumerate(feature_df.columns):
        vifs.append((column, float(variance_inflation_factor(values, i))))
    return pd.DataFrame(vifs, columns=["feature", "vif"]).sort_values("vif", ascending=False)


def run_collinearity(output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = output_dir or REPORTS_DIR
    ensure_dir(output_dir)

    master = build_master()
    numeric = master.select_dtypes(include="number").drop(columns=["SELL_ID", "SELL_CATEGORY"])

    pearson = correlation_matrix(numeric, "pearson")
    spearman = correlation_matrix(numeric, "spearman")
    pearson_path = output_dir / "pearson_corr.csv"
    spearman_path = output_dir / "spearman_corr.csv"
    pearson.to_csv(pearson_path)
    spearman.to_csv(spearman_path)

    top_pairs = top_correlated_pairs(pearson, threshold=0.85)
    top_pairs_path = output_dir / "top_correlations.csv"
    top_pairs.to_csv(top_pairs_path, index=False)

    vif_features = prepare_vif_features(master)
    vif_df = compute_vif(vif_features)
    vif_path = output_dir / "vif.csv"
    vif_df.to_csv(vif_path, index=False)

    return {
        "pearson": pearson_path,
        "spearman": spearman_path,
        "top_pairs": top_pairs_path,
        "vif": vif_path,
    }


if __name__ == "__main__":
    outputs = run_collinearity()
    for name, path in outputs.items():
        print(f"Wrote {name} to {path}")
