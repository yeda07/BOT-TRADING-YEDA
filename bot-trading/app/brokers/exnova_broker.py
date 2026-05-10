from app.brokers.base import BrokerBase, OrderResult, rejected_order


class ExnovaBroker(BrokerBase):
    """Prepared adapter for future demo-only Exnova integration.

    Real trading, browser automation, click automation, safety bypasses and
    terms-of-service workarounds are intentionally not implemented.
    """

    def connect(self) -> None:
        raise NotImplementedError("Exnova adapter is prepared but disabled. Use PaperBroker first.")

    def get_balance(self) -> float:
        raise NotImplementedError("Exnova adapter is prepared but disabled.")

    def get_candles(self, asset: str, timeframe: int, count: int):
        raise NotImplementedError("Exnova candle retrieval is disabled until a compliant demo API is configured.")

    def place_order(self, asset: str, amount: float, direction: str, expiration: int) -> OrderResult:
        return rejected_order(asset, amount, direction, expiration, "Exnova real/demo execution is disabled.")
