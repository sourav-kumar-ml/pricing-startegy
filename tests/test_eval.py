import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.eval import evaluate_model, feature_columns, rolling_origin_cv


def test_feature_columns_excludes_meta():
    df = pd.DataFrame(
        {
            "CALENDAR_DATE": pd.date_range("2020-01-01", periods=2),
            "SELL_ID": [1, 1],
            "log_Q": [0.1, 0.2],
            "f1": [1.0, 2.0],
        }
    )
    feats = feature_columns(df)
    assert "CALENDAR_DATE" not in feats
    assert "SELL_ID" not in feats
    assert "log_Q" not in feats


def test_evaluate_model_returns_metrics_and_residuals():
    df = pd.DataFrame(
        {
            "CALENDAR_DATE": pd.date_range("2020-01-01", periods=3),
            "SELL_ID": [1, 1, 1],
            "log_Q": [0.0, 0.1, 0.2],
            "f1": [1.0, 2.0, 3.0],
        }
    )
    model = LinearRegression().fit(df[["f1"]], df["log_Q"])
    metrics, resid_df = evaluate_model(model, df, split_label="train")
    assert "rmse" in metrics and metrics["rmse"] >= 0
    assert len(resid_df) == len(df)


def test_rolling_origin_cv_returns_folds():
    df = pd.DataFrame(
        {
            "CALENDAR_DATE": pd.date_range("2020-01-01", periods=6, freq="D"),
            "SELL_ID": [1] * 6,
            "log_Q": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "f1": [1, 2, 3, 4, 5, 6],
        }
    )
    folds = rolling_origin_cv(df, n_folds=2)
    assert len(folds) >= 1
