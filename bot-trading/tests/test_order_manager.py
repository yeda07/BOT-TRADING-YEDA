import pandas as pd

from app.brokers.paper_broker import PaperBroker
from app.execution.order_manager import OrderManager
from app.risk.risk_manager import RiskManager


def candle():
    return pd.Series({"timestamp": pd.Timestamp("2026-05-10T12:00:00Z"), "close": 100.0})


def manager(risk_manager=None):
    broker = PaperBroker(initial_balance=1000)
    broker.connect()
    risk = risk_manager or RiskManager(balance=1000, min_model_confidence=0.58)
    return OrderManager(broker, risk, payout=0.87, expiration_candles=1)


def test_order_manager_does_not_trade_hold():
    result = manager().execute_signal("HOLD", 0.9, "EURUSD-OTC", candle())

    assert result["status"] == "HOLD"
    assert result["amount"] == 0.0


def test_order_manager_blocks_low_confidence():
    result = manager().execute_signal("BUY", 0.2, "EURUSD-OTC", candle())

    assert result["status"] == "LOW_CONFIDENCE"


def test_order_manager_blocks_when_risk_manager_disallows_trade():
    risk = RiskManager(balance=1000, max_consecutive_losses=1)
    risk.register_result(-10)

    result = manager(risk).execute_signal("BUY", 0.9, "EURUSD-OTC", candle())

    assert result["status"] == "BLOCKED_BY_RISK"


def test_order_manager_blocks_duplicate_candle():
    order_manager = manager()

    first = order_manager.execute_signal("BUY", 0.9, "EURUSD-OTC", candle())
    second = order_manager.execute_signal("SELL", 0.9, "EURUSD-OTC", candle())

    assert first["status"] == "EXECUTED"
    assert second["status"] == "DUPLICATED_CANDLE"
