from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from app.brokers.demo_broker_base import DemoBrokerBase


class DemoStubBroker(DemoBrokerBase):
    def __init__(self, initial_balance: float = 3000.0, candles: pd.DataFrame | None = None) -> None:
        self.balance = initial_balance
        self.candles = candles.copy() if candles is not None else pd.DataFrame()
        self.connected = False
        self.orders: dict[str, dict] = {}

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_balance(self) -> float:
        return self.balance

    def get_candles(self, asset: str, timeframe_seconds: int, count: int):
        if not self.connected:
            raise RuntimeError("Demo broker is not connected.")
        return self.candles.tail(count).copy()

    def place_demo_order(self, asset: str, amount: float, direction: str, expiration_seconds: int) -> dict:
        if self.get_account_type() != "demo":
            raise RuntimeError("Demo execution blocked because broker account is not demo.")
        if direction not in {"BUY", "SELL"}:
            return {"status": "ERROR", "reason": "Direction must be BUY or SELL."}
        if amount <= 0:
            return {"status": "ERROR", "reason": "Amount must be positive."}
        order_id = str(uuid4())
        order = {
            "order_id": order_id,
            "asset": asset,
            "amount": float(amount),
            "direction": direction,
            "expiration_seconds": int(expiration_seconds),
            "status": "PENDING",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "result": "PENDING",
            "profit": 0.0,
        }
        self.orders[order_id] = order
        return order.copy()

    def get_order_status(self, order_id: str) -> dict:
        if order_id not in self.orders:
            return {"order_id": order_id, "status": "ERROR", "result": "ERROR", "profit": 0.0}
        order = self.orders[order_id]
        if order["status"] == "PENDING":
            won = len(self.orders) % 2 == 1
            profit = order["amount"] * 0.87 if won else -order["amount"]
            self.balance += profit
            order.update({"status": "WON" if won else "LOST", "result": "WON" if won else "LOST", "profit": profit})
        return order.copy()

    def get_account_type(self) -> str:
        return "demo"
