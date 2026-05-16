import sqlite3

from app.storage.trades_repository import TradesRepository


def trade(result: str = "WON", profit: float = 8.7, balance: float = 1008.7, session_id: str = "s1") -> dict:
    return {
        "session_id": session_id,
        "timestamp": "2026-05-10T12:00:00Z",
        "asset": "EURUSD-OTC",
        "signal": "BUY",
        "confidence": 0.7,
        "amount": 10,
        "entry_price": 100,
        "exit_price": 101,
        "result": result,
        "profit": profit,
        "balance": balance,
        "reason": "ok",
        "mode": "paper",
    }


def test_trades_repository_creates_table_and_inserts_trades(tmp_path):
    repository = TradesRepository(str(tmp_path / "trades.db"))

    repository.insert_trade(trade())
    trades = repository.get_all_trades()

    assert len(trades) == 1
    assert trades[0]["asset"] == "EURUSD-OTC"


def test_trades_repository_summary(tmp_path):
    repository = TradesRepository(str(tmp_path / "trades.db"))
    repository.insert_trade(trade("WON", 8.7, 1008.7))
    repository.insert_trade(trade("LOST", -10, 998.7))

    summary = repository.get_summary()

    assert summary["total_trades"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["net_profit"] == -1.3000000000000007
    assert summary["current_balance"] == 998.7


def test_trades_repository_migrates_legacy_table_without_session_id(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE trades (
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
        connection.execute(
            "INSERT INTO trades (timestamp,asset,signal,result,profit,balance,mode) VALUES (?,?,?,?,?,?,?)",
            ("2026-05-10T12:00:00Z", "EURUSD-OTC", "BUY", "WON", 8.7, 1008.7, "paper"),
        )

    repository = TradesRepository(str(db_path))
    trades = repository.get_all_trades()

    assert trades[0]["session_id"] == "legacy"
    assert repository.get_summary("legacy")["net_profit"] == 8.7


def test_trades_repository_summary_filters_by_session(tmp_path):
    repository = TradesRepository(str(tmp_path / "trades.db"))
    repository.insert_trade(trade("WON", 8.7, 1008.7, "s1"))
    repository.insert_trade(trade("LOST", -10, 998.7, "s1"))
    repository.insert_trade(trade("WON", 20, 3020, "s2"))

    historical = repository.get_summary()
    session = repository.get_summary("s1")

    assert historical["total_trades"] == 3
    assert historical["net_profit"] == 18.7
    assert historical["current_balance"] == 3020
    assert session["total_trades"] == 2
    assert session["net_profit"] == -1.3000000000000007
    assert session["current_balance"] == 998.7
    assert session["initial_balance"] == 1000.0


def test_trades_repository_hold_does_not_count_as_executed_trade(tmp_path):
    repository = TradesRepository(str(tmp_path / "trades.db"))
    repository.insert_trade(trade("HOLD", 0, 1000, "s1"))
    repository.insert_trade(trade("WON", 8.7, 1008.7, "s1"))

    summary = repository.get_summary("s1")

    assert summary["total_trades"] == 1
    assert summary["wins"] == 1
    assert summary["net_profit"] == 8.7
