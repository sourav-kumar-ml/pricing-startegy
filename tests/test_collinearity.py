import pandas as pd

from src.eda.collinearity import prepare_vif_features


def test_prepare_vif_features_drops_target_cols():
    df = pd.DataFrame(
        {
            "HOLIDAY": ["Not Holiday", "New Year"],
            "MONTH": [1, 1],
            "PRICE": [10.0, 12.0],
            "QUANTITY": [5, 6],
            "log_Q": [1.6, 1.8],
            "SELL_ID": [1070, 1070],
            "SELL_CATEGORY": [0, 0],
        }
    )
    features = prepare_vif_features(df)
    assert "QUANTITY" not in features.columns
    assert "log_Q" not in features.columns
    assert "SELL_ID" not in features.columns
    assert "SELL_CATEGORY" not in features.columns
