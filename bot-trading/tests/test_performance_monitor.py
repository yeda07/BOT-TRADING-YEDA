from types import SimpleNamespace

from app.monitoring.performance_monitor import PerformanceMonitor
from app.storage.trades_repository import TradesRepository


def test_performance_monitor_detects_daily_loss(tmp_path):
    repo = TradesRepository(str(tmp_path / "trades.db"))
    repo.insert_trade(
        {
            "timestamp": "t",
            "asset": "EURUSD-OTC",
            "signal": "BUY",
            "confidence": 0.9,
            "amount": 100,
            "entry_price": 100,
            "exit_price": 99,
            "result": "LOST",
            "profit": -100,
            "balance": 900,
            "reason": "loss",
            "mode": "demo",
        }
    )
    settings = SimpleNamespace(MAX_DAILY_LOSS=0.05, MAX_CONSECUTIVE_LOSSES=3, MIN_PROFIT_FACTOR=0.0, MIN_TRADES_BEFORE_LIVE=100, PAYOUT=0.87)

    should_stop, reason = PerformanceMonitor(repo, settings).should_stop_trading()

    assert should_stop
    assert "Daily loss" in reason
