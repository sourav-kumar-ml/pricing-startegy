import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.models.elasticity import (
    elasticity_from_results,
    fit_ols,
    LogPScaling,
    price_grid_from_logp,
    predict_curve,
    get_feature_sets,
)


def test_fit_ols_and_elasticity():
    df = pd.DataFrame(
        {
            "log_Q": [0.0, 0.1, 0.2, 0.3],
            "log_p": [1.0, 1.1, 1.2, 1.3],
            "MONTH_2": [0, 1, 0, 1],
        }
    )
    results, feats = fit_ols(df, ["log_p", "MONTH_2"])
    res = elasticity_from_results(1, "full", results, feats)
    assert res.elasticity != 0
    assert res.sell_id == 1
    assert res.model == "full"


def test_predict_curve_shapes():
    results = sm.OLS(
        endog=pd.Series([0.0, 0.1, 0.2, 0.3]),
        exog=sm.add_constant(pd.DataFrame({"log_p": [1.0, 1.1, 1.2, 1.3]}), has_constant="add"),
    ).fit()
    prices = np.array([10.0, 12.0, 14.0])
    base_row = pd.Series({"log_p": np.log(10.0)})
    curve = predict_curve(results, ["log_p"], base_row, prices)
    assert set(["price", "log_p", "qty_pred", "revenue_pred"]).issubset(curve.columns)
    assert len(curve) == len(prices)


def test_get_feature_sets_adds_weather_when_present():
    metadata = {"final_features": ["log_p", "MONTH_2"]}
    df = pd.DataFrame(
        {
            "log_p": [1.0, 1.1],
            "MONTH_2": [0, 1],
            "AVERAGE_TEMPERATURE": [20.0, 22.0],
            "IS_OUTDOOR": [0, 1],
            "TEMP_X_OUTDOOR": [0.0, 22.0],
        }
    )
    sets = get_feature_sets(metadata, df)
    assert "price_weather" in sets
    assert "AVERAGE_TEMPERATURE" in sets["price_weather"]


def test_elasticity_backtransforms_when_log_p_scaled():
    # Create a perfectly log-linear relationship with a known elasticity.
    true_elasticity = 2.0
    log_p = np.array([1.0, 2.0, 3.0, 4.0])
    mean = float(log_p.mean())
    scale = float(log_p.std(ddof=0))
    scaling = LogPScaling(mean=mean, scale=scale)
    scaled_log_p = scaling.scale_log_p(log_p)

    df = pd.DataFrame({"log_Q": true_elasticity * log_p, "log_p": scaled_log_p})
    results, feats = fit_ols(df, ["log_p"])
    res = elasticity_from_results(1, "full", results, feats, log_p_scaling=scaling)
    np.testing.assert_allclose(res.elasticity, true_elasticity, atol=1e-10)


def test_price_grid_from_logp_unscales_prices():
    prices = np.array([10.0, 12.0, 14.0])
    log_p = np.log(prices)
    mean = float(log_p.mean())
    scale = float(log_p.std(ddof=0))
    scaling = LogPScaling(mean=mean, scale=scale)
    scaled_log_p = scaling.scale_log_p(log_p)
    df = pd.DataFrame({"log_p": scaled_log_p})

    grid = price_grid_from_logp(df, num=3, log_p_scaling=scaling)
    np.testing.assert_allclose(grid, prices, rtol=1e-12, atol=1e-12)
