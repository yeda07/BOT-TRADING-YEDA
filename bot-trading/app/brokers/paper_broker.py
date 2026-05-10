import pandas as pd

from app.brokers.base import BrokerBase, OrderResult, accepted_order, rejected_order


class PaperBroker(BrokerBase):
    def __init__(self, initial_balance: float = 1000.0, candles: pd.DataFrame | None = None) -> None:
        self.balance = initial_balance
        self.connected = False
        self.candles = candles.copy() if candles is not None else pd.DataFrame()
        self.orders: list[OrderResult] = []

    def connect(self) -> None:
        self.connected = True

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

        result = accepted_order(asset, amount, direction, expiration, "Paper order accepted.")
        self.orders.append(result)
        return result

    def apply_pnl(self, pnl: float) -> None:
        self.balance += pnl
