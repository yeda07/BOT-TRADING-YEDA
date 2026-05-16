from types import SimpleNamespace

import pandas as pd
import pytest

from app.market.data_feed import CSVDataFeed
from app.market.data_feed_manager import DataFeedManager
from app.market.mock_realtime_feed import MockRealtimeDataFeed


def settings(source, csv_path):
    return SimpleNamespace(
        DATA_FEED_SOURCE=source,
        CANDLES_CSV_PATH=str(csv_path),
        MIN_CANDLES=60,
        ASSET="EURUSD-OTC",
        TIMEFRAME_SECONDS=60,
        LIVE_MAX_STEPS=10,
        FEED_CURSOR_PATH=str(csv_path.parent / "feed_cursor.json"),
        RESET_FEED_CURSOR=False,
        RANDOM_FEED_START=False,
        FEED_START_INDEX=None,
    )


def write_candles(path):
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-10", periods=3, freq="min", tz="UTC"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [1, 1, 1],
        }
    ).to_csv(path, index=False)


def test_data_feed_manager_creates_csv_feed(tmp_path):
    csv_path = tmp_path / "candles.csv"
    write_candles(csv_path)

    feed = DataFeedManager(settings("csv", csv_path)).create_feed()

    assert isinstance(feed, CSVDataFeed)


def test_data_feed_manager_creates_mock_realtime_feed(tmp_path):
    csv_path = tmp_path / "candles.csv"
    write_candles(csv_path)

    feed = DataFeedManager(settings("mock_realtime", csv_path)).create_feed()

    assert isinstance(feed, MockRealtimeDataFeed)


def test_data_feed_manager_rejects_invalid_source(tmp_path):
    with pytest.raises(ValueError, match="Unsupported DATA_FEED_SOURCE"):
        DataFeedManager(settings("bad", tmp_path / "candles.csv")).create_feed()


def test_data_feed_manager_blocks_unimplemented_demo_adapters(tmp_path):
    with pytest.raises(NotImplementedError, match="not fully implemented"):
        DataFeedManager(settings("iqoption_demo", tmp_path / "candles.csv")).create_feed()
    with pytest.raises(NotImplementedError, match="not fully implemented"):
        DataFeedManager(settings("exnova_demo", tmp_path / "candles.csv")).create_feed()
