import pandas as pd
import pytest

from app.brokers.base import BrokerBase
from app.brokers.exnova_broker import ExnovaBroker
from app.brokers.iqoption_broker import IQOptionBroker
from app.brokers.paper_broker import PaperBroker


def sample_candles(rows: int = 5) -> pd.DataFrame:
    prices = [100 + index for index in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": [price + 0.5 for price in prices],
            "volume": [100] * rows,
        }
    )


def test_paper_broker_implements_base_interface():
    broker = PaperBroker(initial_balance=1000, candles=sample_candles())
    assert isinstance(broker, BrokerBase)


def test_paper_broker_requires_connection_for_candles():
    broker = PaperBroker(candles=sample_candles())
    with pytest.raises(RuntimeError, match="not connected"):
        broker.get_candles("EURUSD-OTC", 60, 2)


def test_paper_broker_returns_recent_candles_after_connect():
    broker = PaperBroker(candles=sample_candles(5))
    broker.connect()
    candles = broker.get_candles("EURUSD-OTC", 60, 2)
    assert len(candles) == 2
    assert candles["close"].tolist() == [103.5, 104.5]


def test_paper_broker_accepts_valid_order():
    broker = PaperBroker(initial_balance=1000)
    broker.connect()
    result = broker.place_order("EURUSD-OTC", 10, "BUY", 1)
    assert result.status == "accepted"
    assert result.asset == "EURUSD-OTC"
    assert result.amount == 10
    assert result.direction == "BUY"
    assert result.expiration == 1
    assert len(broker.orders) == 1


def test_paper_broker_rejects_invalid_order():
    broker = PaperBroker(initial_balance=1000)
    broker.connect()
    result = broker.place_order("EURUSD-OTC", 10, "UP", 1)
    assert result.status == "rejected"


def test_paper_broker_resolves_buy_winner():
    broker = PaperBroker(initial_balance=1000)
    broker.connect()
    order = broker.place_order("EURUSD-OTC", 10, "BUY", 1)

    result = broker.resolve_order(order.order_id, entry_price=100, exit_price=101, payout=0.87)

    assert result["status"] == "WON"
    assert result["profit"] == 8.7
    assert broker.get_balance() == 1008.7


def test_paper_broker_resolves_sell_winner():
    broker = PaperBroker(initial_balance=1000)
    broker.connect()
    order = broker.place_order("EURUSD-OTC", 10, "SELL", 1)

    result = broker.resolve_order(order.order_id, entry_price=100, exit_price=99, payout=0.87)

    assert result["status"] == "WON"
    assert result["profit"] == 8.7
    assert broker.get_balance() == 1008.7


def test_real_broker_adapters_do_not_execute_orders():
    iqoption = IQOptionBroker()
    exnova = ExnovaBroker()

    iq_result = iqoption.place_order("EURUSD-OTC", 10, "BUY", 1)
    exnova_result = exnova.place_order("EURUSD-OTC", 10, "SELL", 1)

    assert iq_result.status == "rejected"
    assert exnova_result.status == "rejected"
    with pytest.raises(RuntimeError, match="requires credentials"):
        iqoption.connect()
    with pytest.raises(RuntimeError, match="requires credentials"):
        exnova.connect()


def test_real_broker_demo_execution_not_implemented():
    with pytest.raises(NotImplementedError, match="Authorized demo execution"):
        IQOptionBroker().place_demo_order("EURUSD-OTC", 10, "BUY", 60)
    with pytest.raises(NotImplementedError, match="Authorized demo execution"):
        ExnovaBroker().place_demo_order("EURUSD-OTC", 10, "BUY", 60)
