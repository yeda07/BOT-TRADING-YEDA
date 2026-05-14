import pandas as pd

from app.execution.trade_logger import TradeLogger


def sample_trade() -> dict:
    return {
        "timestamp": "2026-05-10T12:00:00Z",
        "asset": "EURUSD-OTC",
        "signal": "BUY",
        "confidence": 0.7,
        "amount": 10,
        "entry_price": 100,
        "exit_price": 101,
        "result": "WON",
        "profit": 8.7,
        "balance": 1008.7,
        "reason": "ok",
        "mode": "paper",
    }


def test_trade_logger_creates_csv_and_appends_trade(tmp_path):
    log_path = tmp_path / "live_trades.csv"
    logger = TradeLogger(str(log_path))

    logger.log_trade(sample_trade())
    trades = logger.read_trades()

    assert log_path.exists()
    assert len(trades) == 1
    assert trades.iloc[0]["result"] == "WON"
    assert pd.notna(trades.iloc[0]["timestamp"])
