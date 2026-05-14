import time

import pandas as pd

from app.market.data_feed import CSVDataFeed


class MockRealtimeDataFeed(CSVDataFeed):
    def __init__(
        self,
        csv_path: str,
        window_size: int = 300,
        asset: str = "EURUSD-OTC",
        timeframe_seconds: int = 60,
        sleep_seconds: float = 0.0,
        max_steps: int | None = None,
    ):
        super().__init__(
            csv_path=csv_path,
            window_size=window_size,
            asset=asset,
            timeframe_seconds=timeframe_seconds,
            source="mock_realtime",
        )
        self.sleep_seconds = sleep_seconds
        self.max_steps = max_steps
        self._steps = 0

    def get_next_candle(self) -> pd.Series:
        if self.max_steps is not None and self._steps >= self.max_steps:
            raise StopIteration("MockRealtimeDataFeed max_steps reached.")
        candle = super().get_next_candle()
        self._steps += 1
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return candle

    def has_next(self) -> bool:
        if self.max_steps is not None and self._steps >= self.max_steps:
            return False
        return super().has_next()
