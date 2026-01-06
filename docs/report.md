# Pricing Elasticity & Driver Summary

## Overview
- Models: log–log regressions per SKU (OLS + Ridge baselines), plus weather-augmented elasticity variant.
- Data: 2012-01-01 to 2014-09-30 café transactions; features include price (log), holidays, seasonality, day-of-week, and optional weather/outdoor signals.
- Objective: quantify own-price elasticity, visualize price-response/revenue curves, and surface revenue-maximizing price suggestions within observed ranges.

## Elasticities (Own-Price)
- Source: `reports/elasticity.csv`
- Interpretation: elasticity < -1 implies elastic demand; between 0 and -1 implies inelastic.
- Confidence: columns `ci_low`/`ci_high` give approximate intervals; wider intervals indicate more uncertainty.
- Weather robustness: compare `full`, `price_only`, and `price_weather` models; large swings suggest sensitivity to weather covariates.

## Price & Revenue Curves
- Source: `reports/price_curves.csv`
- Construction: price grid spans observed prices; quantities predicted from log–log models; revenue = price * predicted quantity.
- Recommendation summary: `reports/price_recommendations.csv` lists the price that maximizes modeled revenue per SKU/model variant.
- Guidance: treat recommendations as within-sample guidance; avoid extrapolating beyond historical price ranges.

## Key Drivers (qualitative)
- Price (`log_p`): primary driver; magnitude gives own-price elasticity.
- Seasonality: month/day-of-week dummies capture temporal demand shifts; use coefficients for directional insights.
- Holidays: holiday dummies capture spikes/dips; rare holidays collapsed to stabilize estimates.
- Weather (in `price_weather`): temperature/outdoor effects are tested; if elasticity shifts materially, weather plays a role.

## Model Quality & Diagnostics
- Metrics: `reports/eval_metrics.csv` includes RMSE, R², MAPE per SKU/model on train/val.
- Residuals: `reports/eval_residuals.csv` stores residuals with dates for plotting time patterns and heteroskedasticity checks.
- Rolling validation: `reports/eval_rolling_cv.csv` provides forward-in-time folds to gauge stability.
- Heuristics: prefer models with lower val RMSE/MAPE and stable rolling-fold performance.

## Recommendations (How to Use)
- Elasticity: use `reports/elasticity.csv` to decide if demand is elastic; target prices that balance revenue vs. volume.
- Price selection: start from `reports/price_recommendations.csv` but sanity-check against business constraints and observed price bounds.
- Sensitivity: compare `full` vs. `price_weather` elasticities; if similar, price effects are robust to weather; if divergent, consider conditioning on weather.
- Monitoring: re-run pipelines periodically with fresh data; watch residuals for drift.

## Limitations & Assumptions
- Local elasticity: applies near observed prices; extrapolation is unreliable.
- Positive support: logs require positive price/quantity; zeros are excluded.
- Log-linearity: nonlinear price thresholds are not modeled.
- Temporal stationarity: assumes relationships stable over the training window; shifts require re-estimation.
- Multicollinearity: reduced via VIF pruning, but correlated drivers can widen intervals.

## How to Reproduce
- Elasticities/curves/recommendations: `.venv/bin/python -m src.models.elasticity`
- Evaluation metrics/residuals/rolling validation: `.venv/bin/python -m src.models.eval`
- Full test suite: `.venv/bin/pytest`
