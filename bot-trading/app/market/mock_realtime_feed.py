import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.market.data_feed import CSVDataFeed


class EndOfFeedError(Exception):
    pass


class MockRealtimeDataFeed(CSVDataFeed):
    def __init__(
        self,
        csv_path: str,
        window_size: int = 300,
        asset: str = "EURUSD-OTC",
        timeframe_seconds: int = 60,
        sleep_seconds: float = 0.0,
        max_steps: int | None = None,
        cursor_path: str = "data/logs/feed_cursor.json",
        reset_cursor: bool = False,
        random_start: bool = False,
        start_index: int | None = None,
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
        self.cursor_path = Path(cursor_path)
        self.reset_cursor = reset_cursor
        self.random_start = random_start
        self.configured_start_index = start_index
        self.replay_warning = False
        self.start_index = self._resolve_start_index()
        self._cursor = self.start_index
        self.end_index = self.start_index - 1

    def connect(self) -> None:
        if self.reset_cursor:
            self.replay_warning = True
            print("WARNING: Replay mode enabled: feed cursor reset.")
        super().connect()

    def get_next_candle(self) -> pd.Series:
        if self.max_steps is not None and self._steps >= self.max_steps:
            raise StopIteration("MockRealtimeDataFeed max_steps reached.")
        candle = super().get_next_candle()
        self._steps += 1
        self.end_index = self._cursor - 1
        self._write_cursor(candle)
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return candle

    def has_next(self) -> bool:
        if self.max_steps is not None and self._steps >= self.max_steps:
            return False
        return super().has_next()

    def get_status(self) -> dict:
        cursor = self._read_cursor()
        remaining = max(0, len(self._data) - self._cursor)
        return {
            "csv_path": str(self.csv_path),
            "total_candles": len(self._data),
            "cursor_last_index": cursor.get("last_index"),
            "cursor_last_timestamp": cursor.get("last_timestamp"),
            "next_start_index": self._cursor,
            "remaining_candles": remaining,
            "feature_window_size": self.window_size,
            "estimated_usable_candles": max(0, remaining - self.window_size),
        }

    def _resolve_start_index(self) -> int:
        total = len(self._data)
        if self.configured_start_index is not None:
            return self._validated_start_index(int(self.configured_start_index), total)
        if self.random_start:
            lower = min(self.window_size, max(0, total - 1))
            upper = max(lower, total - 1)
            return random.randint(lower, upper) if total else 0
        if self.reset_cursor:
            return 0
        cursor = self._read_cursor()
        if self._cursor_matches(cursor):
            next_index = int(cursor.get("last_index", -1)) + 1
            if next_index >= total:
                raise EndOfFeedError("No remaining candles available. Feed cursor reached the end of the CSV.")
            return self._validated_start_index(next_index, total)
        return 0

    def _validated_start_index(self, index: int, total: int) -> int:
        if index < 0:
            raise ValueError("FEED_START_INDEX must be greater than or equal to 0.")
        if index >= total:
            raise ValueError(f"FEED_START_INDEX {index} is outside CSV range with {total} candles.")
        return index

    def _read_cursor(self) -> dict:
        if not self.cursor_path.exists():
            return {}
        try:
            return json.loads(self.cursor_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_cursor(self, candle: pd.Series) -> None:
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": self.source,
            "csv_path": self._path_key(self.csv_path),
            "asset": self.asset,
            "timeframe_seconds": self.timeframe_seconds,
            "last_index": self.end_index,
            "last_timestamp": str(candle.get("timestamp")),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cursor_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _cursor_matches(self, cursor: dict) -> bool:
        return (
            cursor.get("source") == self.source
            and self._path_key(cursor.get("csv_path", "")) == self._path_key(self.csv_path)
            and cursor.get("asset") == self.asset
            and int(cursor.get("timeframe_seconds", -1)) == int(self.timeframe_seconds)
        )

    def _path_key(self, path) -> str:
        return Path(path).as_posix()
