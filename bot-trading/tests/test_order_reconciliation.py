from app.brokers.demo_stub_broker import DemoStubBroker
from app.execution.order_reconciliation import OrderReconciliation
from app.storage.trades_repository import TradesRepository


def test_order_reconciliation_updates_pending_orders(tmp_path):
    broker = DemoStubBroker(initial_balance=1000)
    broker.connect()
    order = broker.place_demo_order("EURUSD-OTC", 10, "BUY", 60)
    repo = TradesRepository(str(tmp_path / "trades.db"))
    repo.insert_trade(
        {
            "timestamp": "t",
            "asset": "EURUSD-OTC",
            "signal": "BUY",
            "confidence": 0.9,
            "amount": 10,
            "entry_price": 100,
            "exit_price": None,
            "result": "PENDING",
            "profit": 0,
            "balance": 1000,
            "reason": "pending",
            "mode": "demo",
            "order_id": order["order_id"],
        }
    )

    result = OrderReconciliation(broker, repo).reconcile_pending_orders()

    assert result["checked"] == 1
    assert result["updated"] == 1
    assert repo.get_pending_trades() == []
