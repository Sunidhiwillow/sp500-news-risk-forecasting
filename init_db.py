"""
Run this ONCE to bootstrap pipeline.db from what you already have:

  - model_params.json / model_state.json  -- The notebook's output after
    refitting on the combined 2021-2026 dataset
  - a CSV of the full daily history (date, close, return_pct, avg_tone,
    news_count, risk_signal) -- i.e. your notebook's `final_df`, exported
    with `final_df.to_csv("final_df.csv", index=False)`

Usage:
    python init_db.py --params model_params.json --state model_state.json --history final_df.csv
"""

import argparse
import json

import pandas as pd

import db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to model_params.json")
    ap.add_argument("--state", required=True, help="Path to model_state.json")
    ap.add_argument("--history", required=True,
                     help="Path to CSV with columns: Date/date, Close/close, "
                          "Return/return_pct, avg_tone, news_count, risk_signal")
    args = ap.parse_args()

    db.init_db()

    with open(args.params) as f:
        params = json.load(f)
    with open(args.state) as f:
        state = json.load(f)

    db.save_params(params)
    db.save_state(state)

    hist = pd.read_csv(args.history)
    hist.columns = [c.lower() for c in hist.columns]
    rename_map = {"date": "date", "close": "close", "return": "return_pct"}
    hist = hist.rename(columns=rename_map)
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")

    n = 0
    for _, row in hist.iterrows():
        db.upsert_daily_row(
            row["date"], float(row["close"]),
            return_pct=float(row["return_pct"]) if pd.notna(row.get("return_pct")) else None,
            avg_tone=float(row["avg_tone"]) if pd.notna(row.get("avg_tone")) else None,
            news_count=int(row["news_count"]) if pd.notna(row.get("news_count")) else None,
            risk_signal=float(row["risk_signal"]) if pd.notna(row.get("risk_signal")) else None,
        )
        n += 1

    print(f"Bootstrapped {n} historical rows.")
    print(f"model_state.last_date = {state['last_date']}")
    print("Ready -- daily_job.py can now pick up from here.")


if __name__ == "__main__":
    main()
