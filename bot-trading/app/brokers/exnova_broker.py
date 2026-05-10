from app.brokers.base import Broker, Order, OrderResult


class ExnovaBroker(Broker):
    """Placeholder for future demo-only Exnova integration."""

    def connect(self) -> None:
        raise NotImplementedError("Exnova integration is not implemented. Use paper/demo simulation first.")

    def place_order(self, order: Order) -> OrderResult:
        return OrderResult(status="rejected", order=order, message="Exnova broker is not implemented.")

    def get_balance(self) -> float:
        raise NotImplementedError("Exnova integration is not implemented.")

