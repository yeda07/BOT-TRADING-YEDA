from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


Side = Literal["BUY", "SELL"]
OrderStatus = Literal["accepted", "rejected", "filled"]


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    amount: float
    price: float
    timestamp: datetime


@dataclass(frozen=True)
class OrderResult:
    status: OrderStatus
    order: Order
    message: str = ""


class Broker(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> float:
        raise NotImplementedError

