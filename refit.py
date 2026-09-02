"""
Full re-estimation, same logic as the training notebook, run periodically
(weekly) on a rolling window of daily_data.

This is the ONLY place actual model fitting happens in production. It is
cheap: a few hundred to ~1500 daily observations, sub-second on any CPU.
process_new_day() in forecast_engine.py never re-optimizes; it just applies
whatever coefficients this function last produced.

Why refit at all if forecast_engine already walks the recursion forward?
Because omega/alpha/beta/gamma/intercept themselves can drift as the market
regime changes. Refitting periodically keeps them current; the daily
walk-forward keeps the *state* (last residual, last variance) current in
between refits.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import db

ROLLING_WINDOW_DAYS = 365 * 2  # ~2 years of trading days;


def _fit_arimax(df):
    model = ARIMA(df["return_pct"], order=(1, 0, 0), exog=df[["risk_signal_lag1"]])
    fit = model.fit()
    return {
        "intercept": float(fit.params["const"]),
        "ar_l1": float(fit.params["ar.L1"]),
        "risk_signal_lag1": float(fit.params["risk_signal_lag1"]),
    }, fit.resid.values


def _fit_garch(residuals):
    am = arch_model(residuals, vol="Garch", p=1, q=1, dist="t", rescale=False)
    fit = am.fit(disp="off")
    return {
        "omega": float(fit.params["omega"]),
        "alpha": float(fit.params["alpha[1]"]),
        "beta": float(fit.params["beta[1]"]),
    }


def _fit_garchx(residuals, risk_signal_lag1):
    eps = residuals
    x = risk_signal_lag1

    def neg_log_lik(theta):
        omega, alpha, beta, gamma = theta
        n = len(eps)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(eps)
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1] + gamma * x[t - 1]
            sigma2[t] = max(sigma2[t], 1e-10)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps ** 2 / sigma2)
        return -ll

    x0 = [0.01, 0.05, 0.9, 0.01]
    bounds = [(1e-8, None), (0, 1), (0, 1), (None, None)]
    result = minimize(neg_log_lik, x0, bounds=bounds, method="L-BFGS-B")
    omega, alpha, beta, gamma = result.x

    n = len(eps)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(eps)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1] + gamma * x[t - 1]

    return {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)}, sigma2


def run_refit(window_days=ROLLING_WINDOW_DAYS):
    """
    Reads daily_data from the DB, re-fits ARIMAX/GARCH/GARCH-X on the trailing
    `window_days` rows, and overwrites model_params + model_state. Returns
    the new params/state dicts for logging.
    """
    rows = db.get_daily_data(limit_days=window_days)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["return_pct", "risk_signal"]).reset_index(drop=True)
    df["risk_signal_lag1"] = df["risk_signal"].shift(1)
    df = df.dropna(subset=["risk_signal_lag1"]).reset_index(drop=True)

    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} usable rows -- need more history before refitting.")

    arimax_params, arimax_resid = _fit_arimax(df)
    garch_params = _fit_garch(arimax_resid)

    # GARCH's own arch_model recursion, for internal consistency with garch_params
    n = len(arimax_resid)
    sigma2_g = np.zeros(n)
    sigma2_g[0] = np.var(arimax_resid)
    for t in range(1, n):
        sigma2_g[t] = garch_params["omega"] + garch_params["alpha"] * arimax_resid[t - 1] ** 2 \
                      + garch_params["beta"] * sigma2_g[t - 1]

    garchx_params, sigma2_x = _fit_garchx(arimax_resid, df["risk_signal_lag1"].values)

    params = {"arimax": arimax_params, "garch": garch_params, "garchx": garchx_params}
    state = {
        "last_date": df["date"].iloc[-1],
        "last_return": float(df["return_pct"].iloc[-1]),
        "last_residual": float(arimax_resid[-1]),
        "last_risk_signal": float(df["risk_signal_lag1"].iloc[-1]),
        "garch_variance": float(sigma2_g[-1]),
        "garchx_variance": float(sigma2_x[-1]),
    }

    db.save_params(params)
    db.save_state(state)
    return params, state


if __name__ == "__main__":
    p, s = run_refit()
    print("Refit complete.")
    print("params:", p)
    print("state:", s)
