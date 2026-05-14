from abc import ABC, abstractmethod

import pandas as pd


class DataFeedBase(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_latest_candles(self, count: int = 300) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_next_candle(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def has_next(self) -> bool:
        raise NotImplementedError
