from abc import ABC, abstractmethod


class DemoBrokerBase(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_candles(self, asset: str, timeframe_seconds: int, count: int):
        raise NotImplementedError

    @abstractmethod
    def place_demo_order(self, asset: str, amount: float, direction: str, expiration_seconds: int) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_account_type(self) -> str:
        raise NotImplementedError
