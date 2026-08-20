"""
dashboard.py
------------
Read-only view of the DB the pipeline writes to. Run with:
    streamlit run dashboard.py

Auto-refreshes every 5 minutes so it stays current as daily_job.py runs on
its own schedule -- this app never fetches data or fits models itself, it
only reads what the scheduled jobs already computed.
"""

import numpy as np
import pandas as pd
import streamlit as st

import db

st.set_page_config(page_title="S&P 500 Return & Volatility Forecast", layout="wide")

# lightweight auto-refresh without extra dependencies
st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

st.title("S&P 500 -- Daily Return & Volatility Forecast")
st.caption("ARIMAX (mean) + GARCH / GARCH-X (variance), news signal from GDELT. "
           "Volatility forecasts are the model's real strength -- treat the "
           "return forecast as directional context, not a trading signal.")

db.init_db()

try:
    params = db.load_params()
    state = db.load_state()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Model last updated through", state["last_date"])
col2.metric("Last known return", f"{state['last_return']:.3f}%")
col3.metric("Current GARCH-X volatility", f"{state['garchx_variance'] ** 0.5:.3f}")

st.subheader("Latest forecasts vs. realized")
forecasts = db.get_recent_forecasts(n=60)
if forecasts:
    df = pd.DataFrame(forecasts)
    df["garch_vol"] = df["predicted_garch_var"].apply(lambda v: np.sqrt(v) if v is not None else None)
    df["garchx_vol"] = df["predicted_garchx_var"].apply(lambda v: np.sqrt(v) if v is not None else None)

    left, right = st.columns(2)
    with left:
        st.markdown("**Return: predicted vs. actual**")
        st.line_chart(df.set_index("date")[["predicted_return", "actual_return"]])
    with right:
        st.markdown("**Volatility (GARCH vs GARCH-X)**")
        st.line_chart(df.set_index("date")[["garch_vol", "garchx_vol"]])

    st.markdown("**Tomorrow's forecast (as of last close)**")
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted return", f"{latest['forecast_next_return']:.3f}%")
    c2.metric("GARCH volatility", f"{latest['forecast_next_garch_var'] ** 0.5:.3f}")
    c3.metric("GARCH-X volatility", f"{latest['forecast_next_garchx_var'] ** 0.5:.3f}")

    with st.expander("Raw forecast history"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("No forecasts logged yet -- daily_job.py hasn't run since bootstrap.")

with st.expander("Current model parameters"):
    st.json(params)

with st.expander("Current walk-forward state"):
    st.json(state)
