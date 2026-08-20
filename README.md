# News-Driven S&P 500 Return and Volatility Forecasting

A time-series forecasting project investigating whether financial news sentiment and risk signals extracted from **GDELT** can improve forecasts of **S&P 500 returns and volatility**.

The project combines financial time-series models with news-derived features and has been extended from historical model development to a **daily automated forecasting pipeline**.

---

## Overview

Financial markets are influenced not only by historical price movements but also by new information arriving through financial news.

This project investigates whether information extracted from news can provide additional predictive information for:

* **Next-day S&P 500 returns**
* **Next-day S&P 500 volatility**

The project combines:

* S&P 500 market data
* GDELT financial/news data
* News tone and sentiment
* Daily news volume
* A news-based risk signal
* ARIMA / ARIMAX models for returns
* GARCH / GARCH-X models for volatility
* An automated pipeline for daily data collection and forecasting

---

## Project Pipeline

```text
                 GDELT News
                     |
                     v
              Extract News Tone
                     |
                     v
            Daily News Aggregation
                     |
          +----------+----------+
          |                     |
          v                     v
     Average Tone           News Count
          |                     |
          +----------+----------+
                     |
                     v
               Risk Signal
                     |
                     v
             +-------+-------+
             |               |
             v               v
          ARIMAX           GARCH-X
             |               |
             v               v
      Return Forecast   Volatility Forecast
```

The project is now extended with a production-style daily pipeline:

```text
       GDELT News          S&P 500 Data
            |                    |
            v                    v
       News Collection      Market Data
            |                    |
            +---------+----------+
                      |
                      v
              Forecasting Engine
                      |
              +-------+-------+
              |               |
              v               v
        Return Forecast   Volatility Forecast
              |               |
              +-------+-------+
                      |
                      v
                  Dashboard
```

---

## Modeling

### Return Forecasting

ARIMA and ARIMAX models are used to forecast S&P 500 returns.

The ARIMAX model incorporates the lagged news-based risk signal as an exogenous variable.

### Volatility Forecasting

GARCH and GARCH-X models are used to forecast market volatility.

The GARCH-X model extends the traditional GARCH framework by incorporating the news-based risk signal into the volatility dynamics.

The models were initially trained and evaluated using historical data before being integrated into the forecasting pipeline.

---

## Deployment Pipeline

The project is divided into two stages:

### Stage 1 — Model Development

Historical S&P 500 and GDELT data are processed to:

1. Aggregate daily news information
2. Construct the news risk signal
3. Train ARIMAX, GARCH and GARCH-X models
4. Evaluate return and volatility forecasts
5. Save the resulting model parameters and state

### Stage 2 — Automated Forecasting

The trained models are used in a daily pipeline that:

1. Collects new GDELT news data
2. Updates the daily news features and risk signal
3. Retrieves the latest S&P 500 market data
4. Updates the model state
5. Generates next-day return and volatility forecasts
6. Periodically refits the models using updated historical data
7. Stores the latest results for visualization

The deployment stage uses **walk-forward forecasting**, so the production pipeline applies the existing model parameters to new observations rather than retraining the model for every prediction.

---

## Repository Structure

| File                             | Purpose                                                 |
| -------------------------------- | ------------------------------------------------------- |
| `db.py`                          | Database schema and data access utilities               |
| `data_fetch.py`                  | Fetches S&P 500 and GDELT data                          |
| `forecast_engine.py`             | Generates forecasts using the current model state       |
| `refit.py`                       | Periodically refits the forecasting models              |
| `poll_news.py`                   | Collects new GDELT news data                            |
| `daily_job.py`                   | Runs the daily update and forecasting process           |
| `init_db.py`                     | Initializes the database using historical model outputs |
| `dashboard.py`                   | Streamlit dashboard for forecasts and model state       |
| `requirements.txt`               | Dependencies for the dashboard                          |
| `requirements-pipeline.txt`      | Dependencies for the forecasting pipeline               |
| `.github/workflows/schedule.yml` | GitHub Actions workflow for scheduled jobs              |

---

## Data Sources

### S&P 500

Historical and daily market data are obtained using **yfinance**.

### GDELT

Financial and general news data are obtained from the **GDELT Project**.

News tone and other article-level information are aggregated into daily features that are subsequently used to construct the news-based risk signal.

Both data sources are freely accessible and do not require API keys for the current implementation.

---

## Running Locally

Install the required dependencies:

```bash
pip install -r requirements-pipeline.txt
```

Initialize the database using the model outputs from the training stage:

```bash
python init_db.py --params model_params.json --state model_state.json --history final_df.csv
```

Run the news collection:

```bash
python poll_news.py
```

Run the daily forecasting job:

```bash
python daily_job.py
```

Launch the dashboard:

```bash
streamlit run dashboard.py
```

---

## Automation

The forecasting pipeline is scheduled using **GitHub Actions**.

The workflow periodically collects new news data and runs the daily forecasting job. The resulting pipeline state and forecasts are stored in the project's database.

The Streamlit dashboard provides a simple interface for monitoring the latest forecasts and model state.

## Limitations

This project is intended as a **research and forecasting pipeline**, not as a trading system.

Forecast performance depends on the quality and availability of both market and news data. External data sources such as GDELT and yfinance may also change their response formats or availability over time.

The current deployment uses SQLite and GitHub Actions, which is appropriate for the scale of this project but would need to be replaced with a dedicated database and compute environment for a larger production system.

---

## Future Improvements

Potential extensions include:

* Improving the construction of the news risk signal
* Testing additional NLP-based financial sentiment features
* Comparing alternative volatility models
* Incorporating additional market variables
* Improving forecast evaluation with rolling out-of-sample testing
* Migrating the deployment database to PostgreSQL
* Adding model monitoring and automated performance tracking

---

## Tech Stack

**Python · Pandas · NumPy · Statsmodels · ARCH · SciPy · yfinance · GDELT · SQLite · Streamlit · GitHub Actions**
