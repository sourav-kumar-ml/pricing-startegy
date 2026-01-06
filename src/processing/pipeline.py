from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

from src.utils.io import (
    DATE_COLUMN,
    read_date_info,
    read_product_meta,
    read_transactions,
)
from src.utils.paths import CONFIGS_DIR, PROCESSED_DIR, ensure_dir


TARGET_COL = "log_Q"
PRICE_LOG_COL = "log_p"
RARE_HOLIDAY_OTHER = "OtherHoliday"


@dataclass
class TransformationMetadata:
    rows_input: int
    rows_after_filter: int
    dropped_rows_nonpositive: int
    rare_holiday_min_count: int
    dropped_columns: list[str]
    vif_drops: list[dict[str, float | str]]
    final_features: list[str]
    scaled_features: list[str]
    thresholds: Sequence[float]
    features_path: str
    features_with_meta_path: str | None
    scaler_path: str | None
    note: str | None = None

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def collapse_rare_categories(series: pd.Series, min_count: int, other_label: str = RARE_HOLIDAY_OTHER) -> pd.Series:
    counts = series.value_counts(dropna=False)
    rare_labels = counts[counts < min_count].index
    return series.where(~series.isin(rare_labels), other_label)


def load_raw_data():
    date_info = read_date_info()
    transactions = read_transactions()
    product = read_product_meta()
    return date_info, transactions, product


def merge_raw(date_info: pd.DataFrame, transactions: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    date_info = date_info.drop_duplicates(subset=[DATE_COLUMN]).copy()
    date_info["HOLIDAY"] = date_info["HOLIDAY"].replace("NULL", pd.NA)
    product = product.drop_duplicates(subset=["SELL_ID"]).copy()
    merged = transactions.merge(date_info, on=DATE_COLUMN, how="left")
    merged = merged.merge(product, on="SELL_ID", how="left")
    return merged


def engineer_features(df: pd.DataFrame, rare_holiday_min_count: int = 20) -> pd.DataFrame:
    out = df.copy()

    # Clean holiday and collapse rare categories.
    out["HOLIDAY"] = out["HOLIDAY"].fillna("Not Holiday")
    out["HOLIDAY"] = collapse_rare_categories(out["HOLIDAY"], min_count=rare_holiday_min_count)

    # Ensure positive price/quantity for logs.
    non_positive_mask = (out["PRICE"] <= 0) | (out["QUANTITY"] <= 0)
    out = out.loc[~non_positive_mask].reset_index(drop=True)

    out["revenue"] = out["PRICE"] * out["QUANTITY"]
    out[TARGET_COL] = np.log(out["QUANTITY"])
    out[PRICE_LOG_COL] = np.log(out["PRICE"])

    out["MONTH"] = out[DATE_COLUMN].dt.month
    out["DAYOFWEEK"] = out[DATE_COLUMN].dt.dayofweek
    out["IS_MONTH_END"] = out[DATE_COLUMN].dt.is_month_end.astype(int)
    out["YEAR"] = out[DATE_COLUMN].dt.year
    out["TEMP_X_OUTDOOR"] = out["AVERAGE_TEMPERATURE"] * out["IS_OUTDOOR"]
    return out


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["HOLIDAY", "MONTH", "DAYOFWEEK"]
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return encoded


def iterative_vif_prune(
    df: pd.DataFrame, thresholds: Sequence[float], protect: set[str] | frozenset[str] = frozenset()
) -> tuple[pd.DataFrame, list[dict[str, float | str]]]:
    working = df.copy()
    drops: list[dict[str, float | str]] = []
    for threshold in thresholds:
        while working.shape[1] > 1:
            values = working.values
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero encountered")
                vifs = np.array([variance_inflation_factor(values, i) for i in range(values.shape[1])])
            # Sort by VIF descending and drop the highest that is not protected.
            order = np.argsort(vifs)[::-1]
            drop_idx = None
            for idx in order:
                if working.columns[idx] in protect:
                    continue
                if vifs[idx] > threshold:
                    drop_idx = idx
                    break
            if drop_idx is None:
                break
            col_to_drop = working.columns[int(drop_idx)]
            drops.append({"column": col_to_drop, "vif": float(vifs[int(drop_idx)]), "threshold": float(threshold)})
            working = working.drop(columns=[col_to_drop])
    return working, drops


def select_feature_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    drop_columns = [
        "PRICE",
        "QUANTITY",
        "CALENDAR_DATE",
        "ITEM_NAME",
        "ITEM_ID",
        "SELL_ID",
        "SELL_CATEGORY",
        "revenue",
        "YEAR",
        "IS_WEEKEND",  # redundant with DAYOFWEEK dummies
    ]
    # Drop any SELL_CATEGORY suffixes created by merges.
    drop_columns.extend([c for c in df.columns if c.startswith("SELL_CATEGORY_")])

    dropped_present = [c for c in drop_columns if c in df.columns]
    features = df.drop(columns=dropped_present)
    if "HOLIDAY_Not Holiday" in features.columns:
        features = features.drop(columns=["HOLIDAY_Not Holiday"])
        dropped_present.append("HOLIDAY_Not Holiday")
    target = features.pop(TARGET_COL)
    return target, features, dropped_present


def scale_features(df: pd.DataFrame, scale_cols: Iterable[str]) -> tuple[pd.DataFrame, StandardScaler | None]:
    cols = [c for c in scale_cols if c in df.columns]
    if not cols:
        return df, None
    scaler = StandardScaler()
    df.loc[:, cols] = scaler.fit_transform(df[cols])
    return df, scaler


def build_feature_matrix(
    rare_holiday_min_count: int = 20,
    vif_thresholds: Sequence[float] = (10.0, 5.0),
) -> tuple[pd.DataFrame, pd.DataFrame, TransformationMetadata, StandardScaler | None]:
    date_info, transactions, product = load_raw_data()
    merged = merge_raw(date_info, transactions, product)
    rows_input = len(merged)

    engineered = engineer_features(merged, rare_holiday_min_count=rare_holiday_min_count)
    rows_after_filter = len(engineered)
    dropped_rows = rows_input - rows_after_filter

    encoded = encode_features(engineered)
    meta_cols = [c for c in (DATE_COLUMN, "SELL_ID") if c in encoded.columns]
    meta = encoded[meta_cols].copy()
    target, features, dropped_present = select_feature_columns(encoded)

    # Remove zero-variance columns before VIF.
    zero_var_cols = [c for c in features.columns if features[c].nunique() <= 1]
    if zero_var_cols:
        features = features.drop(columns=zero_var_cols)
        dropped_present.extend(zero_var_cols)

    pruned_features, vif_drops = iterative_vif_prune(features, thresholds=vif_thresholds, protect={PRICE_LOG_COL})

    scale_candidates = pruned_features.select_dtypes(include=["float"]).columns.tolist()
    pruned_features, scaler = scale_features(pruned_features, scale_candidates)

    dataset = pd.concat([target.reset_index(drop=True), pruned_features.reset_index(drop=True)], axis=1)
    dataset_with_meta = pd.concat([meta.reset_index(drop=True), dataset], axis=1)

    metadata = TransformationMetadata(
        rows_input=rows_input,
        rows_after_filter=rows_after_filter,
        dropped_rows_nonpositive=dropped_rows,
        rare_holiday_min_count=rare_holiday_min_count,
        dropped_columns=dropped_present,
        vif_drops=vif_drops,
        final_features=list(pruned_features.columns),
        scaled_features=scale_candidates,
        thresholds=vif_thresholds,
        features_path="",
        features_with_meta_path=None,
        scaler_path=None,
        note=None,
    )
    return dataset, dataset_with_meta, metadata, scaler


def save_outputs(
    dataset: pd.DataFrame,
    dataset_with_meta: pd.DataFrame,
    metadata: TransformationMetadata,
    scaler: StandardScaler | None,
    features_path: Path | None = None,
    features_with_meta_path: Path | None = None,
    metadata_path: Path | None = None,
    scaler_path: Path | None = None,
) -> TransformationMetadata:
    ensure_dir(PROCESSED_DIR)
    ensure_dir(CONFIGS_DIR)

    features_path = features_path or (PROCESSED_DIR / "features.parquet")
    features_with_meta_path = features_with_meta_path or (PROCESSED_DIR / "features_with_ids.parquet")
    metadata_path = metadata_path or (CONFIGS_DIR / "transform_metadata.json")
    scaler_path = scaler_path or (CONFIGS_DIR / "scaler.pkl")

    dataset.to_parquet(features_path, index=False)
    metadata.features_path = str(features_path)
    dataset_with_meta.to_parquet(features_with_meta_path, index=False)
    metadata.features_with_meta_path = str(features_with_meta_path)

    if scaler is not None:
        dump(scaler, scaler_path)
        metadata.scaler_path = str(scaler_path)

    metadata.to_json(metadata_path)
    return metadata


def run_pipeline(rare_holiday_min_count: int = 20, vif_thresholds: Sequence[float] = (10.0, 5.0)) -> TransformationMetadata:
    dataset, dataset_with_meta, metadata, scaler = build_feature_matrix(
        rare_holiday_min_count=rare_holiday_min_count, vif_thresholds=vif_thresholds
    )
    metadata = save_outputs(dataset, dataset_with_meta, metadata, scaler)
    return metadata


if __name__ == "__main__":
    meta = run_pipeline()
    print("Wrote features to", meta.features_path)
    if meta.scaler_path:
        print("Saved scaler to", meta.scaler_path)
