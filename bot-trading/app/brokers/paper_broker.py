from app.brokers.base import Broker, Order, OrderResult


class PaperBroker(Broker):
    def __init__(self, initial_balance: float = 1000.0) -> None:
        self.balance = initial_balance
        self.connected = False
        self.orders: list[Order] = []

    def connect(self) -> None:
        self.connected = True

    def place_order(self, order: Order) -> OrderResult:
        if not self.connected:
            return OrderResult(status="rejected", order=order, message="Broker is not connected.")
        if order.amount <= 0:
            return OrderResult(status="rejected", order=order, message="Amount must be positive.")
        if order.amount > self.balance:
            return OrderResult(status="rejected", order=order, message="Insufficient paper balance.")

        self.orders.append(order)
        return OrderResult(status="accepted", order=order, message="Paper order accepted.")

    def get_balance(self) -> float:
        return self.balance

    def apply_pnl(self, pnl: float) -> None:
        self.balance += pnl

