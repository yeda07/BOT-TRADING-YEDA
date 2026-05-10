from app.brokers.base import Broker, Order, OrderResult


class IQOptionBroker(Broker):
    """Placeholder for future demo-only IQ Option integration."""

    def connect(self) -> None:
        raise NotImplementedError("IQ Option integration is not implemented. Use paper/demo simulation first.")

    def place_order(self, order: Order) -> OrderResult:
        return OrderResult(status="rejected", order=order, message="IQ Option broker is not implemented.")

    def get_balance(self) -> float:
        raise NotImplementedError("IQ Option integration is not implemented.")

