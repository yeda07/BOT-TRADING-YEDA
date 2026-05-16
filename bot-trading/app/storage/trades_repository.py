import sqlite3
from pathlib import Path
from datetime import date, datetime


TRADE_FIELDS = [
    "session_id",
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
    "order_id",
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
                    session_id TEXT DEFAULT 'legacy',
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
                    order_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._ensure_column(connection, "session_id", "TEXT DEFAULT 'legacy'")
            self._ensure_column(connection, "order_id", "TEXT")
            connection.execute("UPDATE trades SET session_id='legacy' WHERE session_id IS NULL OR session_id=''")

    def insert_trade(self, trade: dict) -> None:
        payload = dict(trade)
        payload.setdefault("session_id", "legacy")
        values = [_sqlite_value(payload.get(field)) for field in TRADE_FIELDS]
        placeholders = ",".join(["?"] * len(TRADE_FIELDS))
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO trades ({','.join(TRADE_FIELDS)}) VALUES ({placeholders})",
                values,
            )

    def get_all_trades(self, session_id: str | None = None) -> list[dict]:
        with self._connect() as connection:
            if session_id is None:
                rows = connection.execute("SELECT * FROM trades ORDER BY id").fetchall()
            else:
                rows = connection.execute("SELECT * FROM trades WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_pending_trades(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM trades WHERE result='PENDING' ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def update_trade_result(self, order_id: str, result: str, profit: float, balance: float) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE trades SET result=?, profit=?, balance=? WHERE order_id=?",
                (result, profit, balance, order_id),
            )

    def get_summary(self, session_id: str | None = None) -> dict:
        trades = self.get_all_trades(session_id)
        closed = [trade for trade in trades if trade.get("result") in {"WON", "LOST"}]
        wins = sum(1 for trade in closed if trade.get("result") == "WON")
        losses = sum(1 for trade in closed if trade.get("result") == "LOST")
        total = len(closed)
        balances = [float(trade.get("balance")) for trade in trades if trade.get("balance") is not None]
        net_profit = sum(float(trade.get("profit") or 0.0) for trade in closed)
        current_balance = float(balances[-1]) if balances else 0.0
        return {
            "session_id": session_id,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total else 0.0,
            "net_profit": net_profit,
            "current_balance": current_balance,
            "initial_balance": current_balance - net_profit if balances else 0.0,
        }

    def get_last_session_id(self) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id FROM trades WHERE session_id IS NOT NULL AND session_id != '' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["session_id"] if row else None

    def get_sessions_summary(self) -> list[dict]:
        sessions = []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, MIN(timestamp) AS started_at, MAX(timestamp) AS ended_at, MAX(mode) AS mode
                FROM trades
                GROUP BY session_id
                ORDER BY MAX(id)
                """
            ).fetchall()
        for row in rows:
            session_id = row["session_id"] or "legacy"
            summary = self.get_summary(session_id)
            sessions.append({**dict(row), **summary})
        return sessions

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_column(self, connection: sqlite3.Connection, column: str, column_type: str) -> None:
        existing = [row["name"] for row in connection.execute("PRAGMA table_info(trades)").fetchall()]
        if column not in existing:
            connection.execute(f"ALTER TABLE trades ADD COLUMN {column} {column_type}")


def _sqlite_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
