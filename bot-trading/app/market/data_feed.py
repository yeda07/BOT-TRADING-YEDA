from pathlib import Path

import pandas as pd

from app.market.candle_schema import normalize_candles
from app.market.data_feed_base import DataFeedBase


REQUIRED_CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class CSVDataFeed(DataFeedBase):
    def __init__(
        self,
        csv_path: str,
        window_size: int = 300,
        asset: str = "EURUSD-OTC",
        timeframe_seconds: int = 60,
        source: str = "csv",
    ):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        self.csv_path = Path(csv_path)
        self.window_size = window_size
        self.asset = asset
        self.timeframe_seconds = timeframe_seconds
        self.source = source
        self.connected = False
        self._data = self._load_csv()
        self._cursor = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_latest_candles(self, count: int | None = None) -> pd.DataFrame:
        end = self._cursor
        size = count or self.window_size
        start = max(0, end - size)
        return self._data.iloc[start:end].copy()

    def get_next_candle(self) -> pd.Series:
        if not self.has_next():
            raise StopIteration("No more candles available in CSV data feed.")
        candle = self._data.iloc[self._cursor].copy()
        self._cursor += 1
        return candle

    def has_next(self) -> bool:
        return self._cursor < len(self._data)

    def _load_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV candles file not found: {self.csv_path}")

        data = pd.read_csv(self.csv_path)
        missing = [column for column in REQUIRED_CANDLE_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"CSV candles file is missing required columns: {missing}")

        return normalize_candles(data, asset=self.asset, timeframe_seconds=self.timeframe_seconds, source=self.source)
