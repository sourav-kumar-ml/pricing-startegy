import numpy as np
import pandas as pd

from src.processing.pipeline import (
    collapse_rare_categories,
    encode_features,
    engineer_features,
    iterative_vif_prune,
    select_feature_columns,
    scale_features,
)


def make_sample_df():
    return pd.DataFrame(
        {
            "CALENDAR_DATE": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "PRICE": [10.0, 12.0, 11.0],
            "QUANTITY": [5, 6, 7],
            "SELL_ID": [1, 1, 2],
            "SELL_CATEGORY": [0, 0, 1],
            "HOLIDAY": ["New Year", "RareHoliday", "NULL"],
            "IS_WEEKEND": [0, 1, 0],
            "IS_SCHOOLBREAK": [0, 0, 1],
            "AVERAGE_TEMPERATURE": [20.0, 25.0, 18.0],
            "IS_OUTDOOR": [0, 1, 1],
            "ITEM_ID": [101, 102, 103],
            "ITEM_NAME": ["A", "B", "C"],
        }
    )


def test_collapse_rare_categories():
    series = pd.Series(["A", "A", "B", "C"])
    collapsed = collapse_rare_categories(series, min_count=2, other_label="Other")
    assert (collapsed == "Other").sum() == 2  # B and C collapsed


def test_engineer_encode_and_select():
    df = make_sample_df()
    engineered = engineer_features(df, rare_holiday_min_count=2)
    assert "log_Q" in engineered.columns
    assert "log_p" in engineered.columns

    encoded = encode_features(engineered)
    target, features, dropped = select_feature_columns(encoded)

    assert "YEAR" in dropped  # dropped after feature selection
    assert "PRICE" not in features.columns
    assert "QUANTITY" not in features.columns
    assert "log_p" in features.columns
    assert target.shape[0] == features.shape[0]


def test_iterative_vif_prune_removes_collinear_feature():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "z": [1, 0, 1, 0]})
    pruned, drops = iterative_vif_prune(df, thresholds=(5.0,))
    assert len(drops) >= 1
    assert pruned.shape[1] < df.shape[1]


def test_scale_features_zero_mean_unit_var():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0, 1, 0]})
    scaled, scaler = scale_features(df.copy(), ["a"])
    assert scaler is not None
    np.testing.assert_allclose(float(scaled["a"].mean()), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(scaled["a"].std(ddof=0)), 1.0, atol=1e-7)
