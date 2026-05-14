import sqlite3
from pathlib import Path
from datetime import date, datetime


TRADE_FIELDS = [
    "timestamp",
    "asset",
    "signal",
    "confidence",
    "amount",
    "entry_price",
    "exit_price",
    "result",
    "profit",
    "balance",
    "reason",
    "mode",
]


class TradesRepository:
    def __init__(self, db_path: str = "data/logs/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_tables()

    def create_tables(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    asset TEXT,
                    signal TEXT,
                    confidence REAL,
                    amount REAL,
                    entry_price REAL,
                    exit_price REAL,
                    result TEXT,
                    profit REAL,
                    balance REAL,
                    reason TEXT,
                    mode TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def insert_trade(self, trade: dict) -> None:
        values = [_sqlite_value(trade.get(field)) for field in TRADE_FIELDS]
        placeholders = ",".join(["?"] * len(TRADE_FIELDS))
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO trades ({','.join(TRADE_FIELDS)}) VALUES ({placeholders})",
                values,
            )

    def get_all_trades(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM trades ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def get_summary(self) -> dict:
        trades = self.get_all_trades()
        closed = [trade for trade in trades if trade.get("result") in {"WON", "LOST"}]
        wins = sum(1 for trade in closed if trade.get("result") == "WON")
        losses = sum(1 for trade in closed if trade.get("result") == "LOST")
        total = len(closed)
        balances = [trade.get("balance") for trade in trades if trade.get("balance") is not None]
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total else 0.0,
            "net_profit": sum(float(trade.get("profit") or 0.0) for trade in closed),
            "current_balance": float(balances[-1]) if balances else 0.0,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _sqlite_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
