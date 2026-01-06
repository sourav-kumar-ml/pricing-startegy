# Log–Log Pricing Model Assumptions and Limits

- **Local elasticity**: Elasticities from log–log regressions apply locally around observed prices; extrapolating beyond the historical price range is unreliable.
- **Positive prices/quantities only**: Logs require positive values; zeros/negatives are excluded. Interpretations assume strictly positive price/quantity support.
- **Feature inclusion**: Covariates (holidays, seasonality, weather) capture demand shifts; omitting/including them changes elasticity stability. Weather robustness is reported separately.
- **Linearity in logs**: Assumes a log-linear relationship between price and quantity. Nonlinear price effects or thresholds are not captured here.
- **Temporal stationarity**: Assumes relationships are stable over the training window; shifts in customer behavior or product changes may invalidate estimates.
- **No strong multicollinearity**: Remaining features are pruned for VIF, but correlated drivers can still widen confidence intervals.
