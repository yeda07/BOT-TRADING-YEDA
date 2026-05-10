from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import pandas as pd


Direction = Literal["BUY", "SELL"]
OrderStatus = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class OrderResult:
    status: OrderStatus
    asset: str
    amount: float
    direction: str
    expiration: int
    message: str = ""
    timestamp: datetime | None = None


class BrokerBase(ABC):
    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def get_balance(self):
        raise NotImplementedError

    @abstractmethod
    def get_candles(self, asset: str, timeframe: int, count: int):
        raise NotImplementedError

    @abstractmethod
    def place_order(self, asset: str, amount: float, direction: str, expiration: int):
        raise NotImplementedError


def rejected_order(asset: str, amount: float, direction: str, expiration: int, message: str) -> OrderResult:
    return OrderResult(
        status="rejected",
        asset=asset,
        amount=amount,
        direction=direction,
        expiration=expiration,
        message=message,
        timestamp=datetime.now(timezone.utc),
    )


def accepted_order(asset: str, amount: float, direction: str, expiration: int, message: str) -> OrderResult:
    return OrderResult(
        status="accepted",
        asset=asset,
        amount=amount,
        direction=direction,
        expiration=expiration,
        message=message,
        timestamp=datetime.now(timezone.utc),
    )


# Backward-compatible alias for older imports.
Broker = BrokerBase
