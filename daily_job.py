
import datetime
import numpy as np

import db
import data_fetch
import forecast_engine
import refit

REFIT_WEEKDAY = 0  # Monday


def main():
    db.init_db()

    date_str, close_today = data_fetch.fetch_latest_close()

    last_row = db.get_last_daily_row()
    if last_row is None:
        raise RuntimeError(
            "daily_data is empty. Run init_db.py first to bootstrap from your "
            "combined 2021-2026 dataset and notebook-derived params/state."
        )
    if last_row["date"] == date_str:
        print(f"[daily_job] {date_str} already processed, nothing to do.")
        return

    prev_close = last_row["close"]
    return_today = float(np.log(close_today / prev_close) * 100)

    avg_tone, news_count = db.aggregate_news_for_day(date_str)
    if avg_tone is None:
        print(f"[daily_job] WARNING: no news collected for {date_str} "
              f"(check poll_news.py ran during the day). Using risk_signal=0.")
        avg_tone, news_count = 0.0, 0
    risk_signal_today = -avg_tone

    params = db.load_params()
    state = db.load_state()

    new_state, result = forecast_engine.process_new_day(
        params, state, return_today, risk_signal_today, date_str
    )

    db.save_state(new_state)
    db.upsert_daily_row(
        date_str, close_today, return_pct=return_today,
        avg_tone=avg_tone, news_count=news_count, risk_signal=risk_signal_today,
    )
    db.save_forecast_row(date_str, result["realized_forecast"], return_today, result["forecast_next"])
    db.clear_news_for_day(date_str)  # already aggregated into daily_data

    print(f"[daily_job] {date_str}: close={close_today:.2f} return={return_today:.3f}% "
          f"risk_signal={risk_signal_today:.3f} news_count={news_count}")
    print(f"[daily_job] forecast for tomorrow: return={result['forecast_next']['return']:.3f}% "
          f"garch_vol={result['forecast_next']['garch_var'] ** 0.5:.3f} "
          f"garchx_vol={result['forecast_next']['garchx_var'] ** 0.5:.3f}")

    today = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    if today.weekday() == REFIT_WEEKDAY:
        print("[daily_job] refit day -- running full re-estimation")
        refit.run_refit()


if __name__ == "__main__":
    main()
