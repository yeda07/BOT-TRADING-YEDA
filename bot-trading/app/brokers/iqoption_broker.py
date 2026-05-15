from app.brokers.base import BrokerBase, OrderResult, rejected_order
from app.brokers.demo_broker_base import DemoBrokerBase


class IQOptionBroker(BrokerBase, DemoBrokerBase):
    """Prepared adapter for future demo-only IQ Option integration.

    Real trading, browser automation, click automation, safety bypasses and
    terms-of-service workarounds are intentionally not implemented.
    """

    def __init__(self, email: str | None = None, password: str | None = None) -> None:
        self.email = email
        self.password = password
        self.connected = False

    @staticmethod
    def adapter_not_implemented_message() -> str:
        return "Data feed source selected but adapter is not fully implemented yet."

    def connect(self) -> None:
        if not self.email or not self.password:
            raise RuntimeError("IQ Option demo data feed requires credentials in .env.")
        raise NotImplementedError(self.adapter_not_implemented_message())

    def disconnect(self) -> None:
        self.connected = False

    def get_balance(self) -> float:
        raise NotImplementedError("IQ Option adapter is prepared but disabled.")

    def get_candles(self, asset: str, timeframe_seconds: int, count: int):
        raise NotImplementedError(self.adapter_not_implemented_message())

    def place_order(self, asset: str, amount: float, direction: str, expiration: int) -> OrderResult:
        return rejected_order(asset, amount, direction, expiration, "Real broker execution is not enabled.")

    def place_demo_order(self, asset: str, amount: float, direction: str, expiration_seconds: int) -> dict:
        raise NotImplementedError("Authorized demo execution is not implemented for this broker.")

    def get_order_status(self, order_id: str) -> dict:
        raise NotImplementedError("Authorized demo execution is not implemented for this broker.")

    def get_account_type(self) -> str:
        return "demo"
