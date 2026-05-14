from app.storage.trades_repository import TradesRepository


def trade(result: str = "WON", profit: float = 8.7, balance: float = 1008.7) -> dict:
    return {
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
