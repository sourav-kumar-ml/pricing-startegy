import pandas as pd

from src.models.splits import time_based_split


def test_time_based_split_respects_order():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "CALENDAR_DATE": dates[::-1],  # reverse order
            "SELL_ID": [1] * 5,
            "log_Q": range(5),
        }
    )
    train, val = time_based_split(df, ratio=0.6)
    assert train["CALENDAR_DATE"].is_monotonic_increasing
    assert val["CALENDAR_DATE"].is_monotonic_increasing
    # 5 rows -> split index 3
    assert len(train) == 3
    assert len(val) == 2
