import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy TEXT NOT NULL DEFAULT '',
    signal TEXT NOT NULL,
    confidence REAL NOT NULL,
    entry REAL NOT NULL,
    exit REAL NOT NULL,
    stake REAL NOT NULL,
    pnl REAL NOT NULL,
    balance REAL NOT NULL,
    reason TEXT NOT NULL
);
"""


def connect_sqlite(path: str | Path = "data/logs/trades.db") -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    if "strategy" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN strategy TEXT NOT NULL DEFAULT ''")
    conn.commit()
    return conn


def save_trades(trades: pd.DataFrame, path: str | Path = "data/logs/trades.db") -> int:
    if trades.empty:
        return 0
    conn = connect_sqlite(path)
    try:
        trades.to_sql("trades", conn, if_exists="append", index=False)
        return len(trades)
    finally:
        conn.close()
