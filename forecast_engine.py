"""
Applies the fitted ARIMAX / GARCH / GARCH-X coefficients one trading day at
a time, without ever re-optimizing. This is the "cheap, instant" path that
runs on every new close price. Full re-estimation lives in refit.py and runs
on a much slower cadence (see daily_job.py).

Mirrors the recursions in your training notebook exactly:
  ARIMAX(1,0,0):  Return_t = intercept + ar_l1 * Return_{t-1} + beta_risk * risk_signal_lag1_t
  GARCH(1,1):     sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
  GARCH-X(1,1):   sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1} + gamma * risk_signal_lag1_t

`state` always represents "everything known as of the last finalized trading
day" -- last_return, last_residual, last_risk_signal, garch_variance,
garchx_variance. process_new_day() consumes one new day's actual data and
returns (a) the forecast that WAS made for that day using only prior info
(so you can score accuracy), (b) the forecast for the day after it (what you
show on the dashboard as "tomorrow's prediction"), and (c) the updated state
to persist.
"""


def _arimax_predict(params, prev_return, prev_risk_signal):
    p = params["arimax"]
    return p["intercept"] + p["ar_l1"] * prev_return + p["risk_signal_lag1"] * prev_risk_signal


def _garch_variance(params, prev_eps_sq, prev_variance):
    p = params["garch"]
    return p["omega"] + p["alpha"] * prev_eps_sq + p["beta"] * prev_variance


def _garchx_variance(params, prev_eps_sq, prev_variance, prev_risk_signal):
    p = params["garchx"]
    return (p["omega"] + p["alpha"] * prev_eps_sq
            + p["beta"] * prev_variance + p["gamma"] * prev_risk_signal)


def process_new_day(params: dict, state: dict, actual_return: float,
                     actual_risk_signal: float, date_str: str):
    """
    actual_return       -- today's realized log return * 100 (news/close now known)
    actual_risk_signal  -- today's risk_signal (-avg_tone), NOT lagged
    date_str             -- today's date, becomes the new state's last_date

    Returns (new_state, result) where result contains:
      realized_forecast -- what the model predicted FOR today, made using
                            only yesterday's info (compare to actual_return
                            to score accuracy)
      forecast_next     -- the model's prediction for the NEXT trading day,
                            what the dashboard should show as "tomorrow"
    """
    prev_return = state["last_return"]
    prev_residual = state["last_residual"]
    prev_risk_signal = state["last_risk_signal"]
    prev_garch_var = state["garch_variance"]
    prev_garchx_var = state["garchx_variance"]

    # 1. What the model predicted for TODAY, using only info through yesterday
    predicted_return_today = _arimax_predict(params, prev_return, prev_risk_signal)
    residual_today = actual_return - predicted_return_today

    garch_var_today = _garch_variance(params, prev_residual ** 2, prev_garch_var)
    garchx_var_today = _garchx_variance(params, prev_residual ** 2, prev_garchx_var, prev_risk_signal)

    realized_forecast = {
        "return": predicted_return_today,
        "garch_var": garch_var_today,
        "garchx_var": garchx_var_today,
    }

    # 2. Roll state forward to today
    new_state = {
        "last_date": date_str,
        "last_return": actual_return,
        "last_residual": residual_today,
        "last_risk_signal": actual_risk_signal,
        "garch_variance": garch_var_today,
        "garchx_variance": garchx_var_today,
    }

    # 3. Forecast for the NEXT trading day (this is "tomorrow's prediction")
    forecast_next_return = _arimax_predict(params, actual_return, actual_risk_signal)
    forecast_next_garch_var = _garch_variance(params, residual_today ** 2, garch_var_today)
    forecast_next_garchx_var = _garchx_variance(params, residual_today ** 2, garchx_var_today, actual_risk_signal)

    forecast_next = {
        "return": forecast_next_return,
        "garch_var": forecast_next_garch_var,
        "garchx_var": forecast_next_garchx_var,
    }

    return new_state, {"realized_forecast": realized_forecast, "forecast_next": forecast_next}
