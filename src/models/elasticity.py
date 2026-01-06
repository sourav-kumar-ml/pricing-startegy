from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.io import read_date_info

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR, CONFIGS_DIR, ensure_dir


DATE_COLUMN = "CALENDAR_DATE"
SELL_ID = "SELL_ID"
TARGET_COLUMN = "log_Q"
PRICE_LOG_COL = "log_p"


@dataclass
class ElasticityResult:
    sell_id: int
    model: str
    elasticity: float
    ci_low: float | None
    ci_high: float | None


def load_metadata(path: Path | None = None) -> dict:
    path = path or (CONFIGS_DIR / "transform_metadata.json")
    return json.loads(Path(path).read_text())


def load_features_with_ids(metadata: dict) -> pd.DataFrame:
    features_path = metadata.get("features_with_meta_path") or metadata.get("features_with_ids_path")
    if not features_path:
        raise ValueError("features_with_meta_path missing in transform_metadata.json")
    df = pd.read_parquet(features_path)
    expected = {SELL_ID, DATE_COLUMN, TARGET_COLUMN}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")
    return df


def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Attach weather/outdoor features for robustness runs."""
    date_info = read_date_info().drop_duplicates(subset=[DATE_COLUMN])
    merged = df.merge(
        date_info[[DATE_COLUMN, "AVERAGE_TEMPERATURE", "IS_OUTDOOR"]],
        on=DATE_COLUMN,
        how="left",
    )
    merged["TEMP_X_OUTDOOR"] = merged["AVERAGE_TEMPERATURE"] * merged["IS_OUTDOOR"]
    return merged


def get_feature_sets(metadata: dict, df: pd.DataFrame) -> Dict[str, List[str]]:
    final_features: List[str] = metadata.get("final_features", [])
    if PRICE_LOG_COL not in final_features:
        final_features = [PRICE_LOG_COL] + final_features
    base = [f for f in final_features if f in df.columns]
    sets = {
        "full": base,
        "price_only": [PRICE_LOG_COL],
    }
    weather_feats = [PRICE_LOG_COL, "AVERAGE_TEMPERATURE", "IS_OUTDOOR", "TEMP_X_OUTDOOR"]
    weather_present = [f for f in weather_feats if f in df.columns]
    if PRICE_LOG_COL in weather_present and len(weather_present) > 1:
        sets["price_weather"] = weather_present
    return sets


def fit_ols(df: pd.DataFrame, features: List[str]) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, List[str]]:
    X = df[features]
    y = df[TARGET_COLUMN]
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X)
    results = model.fit()
    return results, X.columns.tolist()


def elasticity_from_results(
    sell_id: int, model_name: str, results: sm.regression.linear_model.RegressionResultsWrapper, feature_names: List[str]
) -> ElasticityResult:
    if PRICE_LOG_COL not in feature_names:
        raise ValueError(f"{PRICE_LOG_COL} missing from model features")
    coef = float(results.params[PRICE_LOG_COL])
    ci_low = ci_high = None
    try:
        ci = results.conf_int().loc[PRICE_LOG_COL]
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except Exception:
        ci_low = ci_high = None
    return ElasticityResult(
        sell_id=sell_id,
        model=model_name,
        elasticity=coef,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def price_grid_from_logp(df: pd.DataFrame, num: int = 25) -> np.ndarray:
    prices = np.exp(df[PRICE_LOG_COL])
    p_min, p_max = prices.min(), prices.max()
    if p_min <= 0:
        p_min = prices[prices > 0].min()
    return np.linspace(p_min, p_max, num=num)


def build_curve_inputs(
    base_row: pd.Series, prices: np.ndarray, features: List[str]
) -> pd.DataFrame:
    grid = pd.DataFrame({PRICE_LOG_COL: np.log(prices)})
    # replicate base values for non-price features
    for feat in features:
        if feat in (PRICE_LOG_COL,):
            continue
        grid[feat] = base_row.get(feat, 0)
    return grid[features]


def predict_curve(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    features: List[str],
    base_row: pd.Series,
    prices: np.ndarray,
) -> pd.DataFrame:
    clean_feats = [f for f in features if f != "const"]
    X_grid = build_curve_inputs(base_row, prices, clean_feats)
    X_grid = sm.add_constant(X_grid, has_constant="add")
    exog_names = list(results.params.index)  # aligns with param vector length/order
    for col in exog_names:
        if col not in X_grid.columns:
            X_grid[col] = 0
    X_grid = X_grid[exog_names]
    log_q_pred = results.predict(X_grid)
    qty_pred = np.exp(log_q_pred)
    revenue = prices * qty_pred
    return pd.DataFrame({"price": prices, "log_p": np.log(prices), "qty_pred": qty_pred, "revenue_pred": revenue})


def summarize_revenue(curves: pd.DataFrame) -> pd.DataFrame:
    idx = curves.groupby(["SELL_ID", "model"])["revenue_pred"].idxmax()
    return curves.loc[idx, ["SELL_ID", "model", "price", "qty_pred", "revenue_pred"]].rename(
        columns={"price": "price_at_max_revenue", "qty_pred": "qty_at_max_revenue", "revenue_pred": "max_revenue"}
    )


def run_elasticity(num_price_points: int = 25) -> dict[str, Path]:
    metadata = load_metadata()
    df = load_features_with_ids(metadata)
    df = attach_weather(df)
    feature_sets = get_feature_sets(metadata, df)

    elasticity_rows: List[dict] = []
    curve_rows: List[dict] = []
    for sell_id, group in df.groupby(SELL_ID):
        base_row = group.median(numeric_only=True)
        price_grid = price_grid_from_logp(group, num=num_price_points)
        for model_name, feats in feature_sets.items():
            # drop missing feature columns if not present (e.g., dummy absence)
            present_feats = [f for f in feats if f in group.columns]
            if PRICE_LOG_COL not in present_feats:
                continue
            results, used_features = fit_ols(group, present_feats)
            elasticity = elasticity_from_results(int(sell_id), model_name, results, used_features)
            elasticity_rows.append(
                {
                    "SELL_ID": elasticity.sell_id,
                    "model": elasticity.model,
                    "elasticity": elasticity.elasticity,
                    "ci_low": elasticity.ci_low,
                    "ci_high": elasticity.ci_high,
                }
            )
            curve = predict_curve(results, used_features, base_row, price_grid)
            curve["SELL_ID"] = int(sell_id)
            curve["model"] = model_name
            curve_rows.append(curve)

    elasticity_df = pd.DataFrame(elasticity_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    ensure_dir(REPORTS_DIR)
    elasticity_path = REPORTS_DIR / "elasticity.csv"
    curves_path = REPORTS_DIR / "price_curves.csv"
    recommendations_path = REPORTS_DIR / "price_recommendations.csv"
    elasticity_df.to_csv(elasticity_path, index=False)
    if not curve_df.empty:
        curve_df.to_csv(curves_path, index=False)
        summarize_revenue(curve_df).to_csv(recommendations_path, index=False)
    return {"elasticity": elasticity_path, "curves": curves_path, "recommendations": recommendations_path}


if __name__ == "__main__":
    paths = run_elasticity()
    for name, path in paths.items():
        print(f"Wrote {name} to {path}")
