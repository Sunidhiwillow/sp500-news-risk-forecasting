import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "pipeline.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_data (
    date        TEXT PRIMARY KEY,   -- ISO date, e.g. 2026-08-17
    close       REAL NOT NULL,
    return_pct  REAL,               -- log return * 100, NULL for first row
    avg_tone    REAL,
    news_count  INTEGER,
    risk_signal REAL                -- -avg_tone
);

CREATE TABLE IF NOT EXISTS news_raw (
    doc_id      TEXT PRIMARY KEY,   -- GDELT DocumentIdentifier, used for dedup
    day         TEXT NOT NULL,      -- calendar date this record belongs to
    tone        REAL NOT NULL,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_params (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    params_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_state (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    state_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts (
    date                    TEXT PRIMARY KEY,
    predicted_return        REAL,
    predicted_garch_var     REAL,
    predicted_garchx_var    REAL,
    actual_return           REAL,
    forecast_next_return    REAL,
    forecast_next_garch_var REAL,
    forecast_next_garchx_var REAL,
    created_at              TEXT NOT NULL
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


# ---------- daily_data ----------

def upsert_daily_row(date, close, return_pct=None, avg_tone=None, news_count=None, risk_signal=None):
    conn = get_conn()
    conn.execute(
        """INSERT INTO daily_data (date, close, return_pct, avg_tone, news_count, risk_signal)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(date) DO UPDATE SET
             close=excluded.close,
             return_pct=COALESCE(excluded.return_pct, daily_data.return_pct),
             avg_tone=COALESCE(excluded.avg_tone, daily_data.avg_tone),
             news_count=COALESCE(excluded.news_count, daily_data.news_count),
             risk_signal=COALESCE(excluded.risk_signal, daily_data.risk_signal)
        """,
        (date, close, return_pct, avg_tone, news_count, risk_signal),
    )
    conn.commit()
    conn.close()


def get_daily_data(limit_days=None):
    """Returns full daily_data history as a list of dict rows, oldest first."""
    conn = get_conn()
    q = "SELECT * FROM daily_data ORDER BY date ASC"
    rows = [dict(r) for r in conn.execute(q).fetchall()]
    conn.close()
    if limit_days:
        rows = rows[-limit_days:]
    return rows


def get_last_daily_row():
    conn = get_conn()
    row = conn.execute("SELECT * FROM daily_data ORDER BY date DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


# ---------- news_raw ----------

def insert_news_records(records):
    """records: list of dicts with keys doc_id, day, tone, fetched_at. Silently skips dupes."""
    if not records:
        return 0
    conn = get_conn()
    inserted = 0
    for r in records:
        try:
            conn.execute(
                "INSERT INTO news_raw (doc_id, day, tone, fetched_at) VALUES (?, ?, ?, ?)",
                (r["doc_id"], r["day"], r["tone"], r["fetched_at"]),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # already have this document
    conn.commit()
    conn.close()
    return inserted


def aggregate_news_for_day(day):
    """Returns (avg_tone, news_count) for a given calendar day from news_raw."""
    conn = get_conn()
    row = conn.execute(
        "SELECT AVG(tone) as avg_tone, COUNT(*) as news_count FROM news_raw WHERE day = ?",
        (day,),
    ).fetchone()
    conn.close()
    if row is None or row["news_count"] == 0:
        return None, 0
    return row["avg_tone"], row["news_count"]


def clear_news_for_day(day):
    conn = get_conn()
    conn.execute("DELETE FROM news_raw WHERE day = ?", (day,))
    conn.commit()
    conn.close()


# ---------- model_params / model_state ----------

def save_params(params: dict):
    conn = get_conn()
    conn.execute(
        """INSERT INTO model_params (id, params_json, updated_at) VALUES (1, ?, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET params_json=excluded.params_json, updated_at=datetime('now')""",
        (json.dumps(params),),
    )
    conn.commit()
    conn.close()


def load_params() -> dict:
    conn = get_conn()
    row = conn.execute("SELECT params_json FROM model_params WHERE id = 1").fetchone()
    conn.close()
    if row is None:
        raise RuntimeError("No model_params found. Run init_db.py to bootstrap from your notebook output.")
    return json.loads(row["params_json"])


def save_state(state: dict):
    conn = get_conn()
    conn.execute(
        """INSERT INTO model_state (id, state_json, updated_at) VALUES (1, ?, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json, updated_at=datetime('now')""",
        (json.dumps(state),),
    )
    conn.commit()
    conn.close()


def load_state() -> dict:
    conn = get_conn()
    row = conn.execute("SELECT state_json FROM model_state WHERE id = 1").fetchone()
    conn.close()
    if row is None:
        raise RuntimeError("No model_state found. Run init_db.py to bootstrap from your notebook output.")
    return json.loads(row["state_json"])


# ---------- forecasts ----------

def save_forecast_row(date, predicted, actual_return, forecast_next):
    conn = get_conn()
    conn.execute(
        """INSERT INTO forecasts
           (date, predicted_return, predicted_garch_var, predicted_garchx_var,
            actual_return, forecast_next_return, forecast_next_garch_var, forecast_next_garchx_var, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(date) DO UPDATE SET
             predicted_return=excluded.predicted_return,
             predicted_garch_var=excluded.predicted_garch_var,
             predicted_garchx_var=excluded.predicted_garchx_var,
             actual_return=excluded.actual_return,
             forecast_next_return=excluded.forecast_next_return,
             forecast_next_garch_var=excluded.forecast_next_garch_var,
             forecast_next_garchx_var=excluded.forecast_next_garchx_var
        """,
        (
            date,
            predicted["return"], predicted["garch_var"], predicted["garchx_var"],
            actual_return,
            forecast_next["return"], forecast_next["garch_var"], forecast_next["garchx_var"],
        ),
    )
    conn.commit()
    conn.close()


def get_recent_forecasts(n=60):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM forecasts ORDER BY date DESC LIMIT ?", (n,)).fetchall()
    conn.close()
    return [dict(r) for r in rows][::-1]
