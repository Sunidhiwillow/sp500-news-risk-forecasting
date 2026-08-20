
import io
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GKG 2.1 field layout (0-indexed) -- only the columns we need
GKG_COL_DATE = 1
GKG_COL_DOC_ID = 4
GKG_COL_V2TONE = 15


def fetch_latest_close(ticker="^GSPC"):
    """
    Returns (date_str, close_price) for the most recent completed trading day.
    Uses a 5-day lookback window so it still works right after a holiday/weekend.
    """
    data = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    last_date = close.index[-1]
    return last_date.strftime("%Y-%m-%d"), float(close.iloc[-1])


def fetch_latest_gkg_tone():
    """
    Downloads the latest GDELT GKG 15-min export and returns a list of dicts:
      {doc_id, day, tone, fetched_at}
    ready to hand to db.insert_news_records(). Returns [] on any transient
    failure so a single missed poll never crashes the scheduled job.
    """
    try:
        lastupdate = requests.get(GDELT_LASTUPDATE_URL, timeout=15).text.strip().splitlines()
        gkg_url = next(line.split(" ")[-1] for line in lastupdate if line.endswith(".gkg.csv.zip"))

        resp = requests.get(gkg_url, timeout=30)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inner_name = zf.namelist()[0]
            raw = zf.read(inner_name)

        df = pd.read_csv(
            io.BytesIO(raw), sep="\t", header=None, dtype=str,
            usecols=[GKG_COL_DATE, GKG_COL_DOC_ID, GKG_COL_V2TONE],
            names=None, engine="python", on_bad_lines="skip",
        )
        df.columns = ["gkg_date", "doc_id", "v2tone"]

        df["tone"] = df["v2tone"].str.split(",").str[0].astype(float)
        df["day"] = pd.to_datetime(df["gkg_date"], format="%Y%m%d%H%M%S").dt.strftime("%Y-%m-%d")

        fetched_at = datetime.now(timezone.utc).isoformat()
        records = [
            {"doc_id": row.doc_id, "day": row.day, "tone": row.tone, "fetched_at": fetched_at}
            for row in df.itertuples()
            if pd.notna(row.tone)
        ]
        return records

    except Exception as e:
        print(f"[data_fetch] GDELT poll failed, skipping this cycle: {e}")
        return []
