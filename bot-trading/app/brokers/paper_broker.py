import pandas as pd
from datetime import datetime, timezone
from uuid import uuid4

from app.brokers.base import BrokerBase, OrderResult, rejected_order


class PaperBroker(BrokerBase):
    def __init__(self, initial_balance: float = 1000.0, candles: pd.DataFrame | None = None) -> None:
        self.balance = initial_balance
        self.connected = False
        self.candles = candles.copy() if candles is not None else pd.DataFrame()
        self.orders: list[OrderResult] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_balance(self) -> float:
        return self.balance

    def get_candles(self, asset: str, timeframe: int, count: int) -> pd.DataFrame:
        if not self.connected:
            raise RuntimeError("Broker is not connected.")
        if count <= 0:
            raise ValueError("count must be positive.")
        return self.candles.tail(count).copy()

    def place_order(self, asset: str, amount: float, direction: str, expiration: int) -> OrderResult:
        if not self.connected:
            return rejected_order(asset, amount, direction, expiration, "Broker is not connected.")
        if direction not in {"BUY", "SELL"}:
            return rejected_order(asset, amount, direction, expiration, "Direction must be BUY or SELL.")
        if amount <= 0:
            return rejected_order(asset, amount, direction, expiration, "Amount must be positive.")
        if expiration < 1:
            return rejected_order(asset, amount, direction, expiration, "Expiration must be at least 1 candle.")
        if amount > self.balance:
            return rejected_order(asset, amount, direction, expiration, "Insufficient paper balance.")

        now = datetime.now(timezone.utc)
        result = OrderResult(
            status="accepted",
            asset=asset,
            amount=amount,
            direction=direction,
            expiration=expiration,
            message="Paper order accepted.",
            timestamp=now,
            order_id=str(uuid4()),
            entry_time=now,
            profit=0.0,
        )
        self.orders.append(result)
        return result

    def resolve_order(self, order_id: str, entry_price: float, exit_price: float, payout: float) -> dict:
        order = next((item for item in self.orders if item.order_id == order_id), None)
        if order is None:
            raise ValueError(f"Order not found: {order_id}")
        if payout <= 0:
            raise ValueError("payout must be positive.")

        won = (order.direction == "BUY" and exit_price > entry_price) or (
            order.direction == "SELL" and exit_price < entry_price
        )
        profit = order.amount * payout if won else -order.amount
        self.balance += profit
        return {
            "order_id": order.order_id,
            "asset": order.asset,
            "amount": order.amount,
            "direction": order.direction,
            "expiration": order.expiration,
            "entry_time": order.entry_time,
            "status": "WON" if won else "LOST",
            "profit": profit,
            "balance": self.balance,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
        }

    def apply_pnl(self, pnl: float) -> None:
        self.balance += pnl
