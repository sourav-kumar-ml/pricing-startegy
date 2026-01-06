import pandas as pd
from src.models.baseline import feature_columns, train_and_eval


def test_feature_columns_excludes_ids_and_target():
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
    assert "f1" in feats


def test_train_and_eval_runs():
    df = pd.DataFrame(
        {
            "CALENDAR_DATE": pd.date_range("2020-01-01", periods=4),
            "SELL_ID": [1, 1, 1, 1],
            "log_Q": [0.0, 0.1, 0.2, 0.3],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 3.0, 4.0, 5.0],
        }
    )
    train = df.iloc[:3]
    val = df.iloc[3:]
    metrics = train_and_eval(train, val)
    assert "ols" in metrics and "ridge" in metrics
    for model_name, m in metrics.items():
        assert m["train_rmse"] >= 0
        assert m["val_rmse"] >= 0
        assert "f1" in m["coefficients"]
        assert "f2" in m["coefficients"]
