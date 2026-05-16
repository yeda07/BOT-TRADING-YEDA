import pandas as pd

from app.market.mock_realtime_feed import MockRealtimeDataFeed


def write_candles(path, rows: int = 5):
    prices = [100 + index for index in range(rows)]
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-10", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": prices,
            "volume": [10] * rows,
        }
    ).to_csv(path, index=False)


def test_mock_realtime_feed_does_not_look_ahead(tmp_path):
    csv_path = tmp_path / "candles.csv"
    write_candles(csv_path)
    feed = MockRealtimeDataFeed(str(csv_path), window_size=3, max_steps=2, cursor_path=str(tmp_path / "feed_cursor.json"))

    feed.connect()
    first = feed.get_next_candle()
    latest = feed.get_latest_candles(10)

    assert len(latest) == 1
    assert latest.iloc[-1]["timestamp"] == first["timestamp"]
    assert feed.has_next()
    feed.get_next_candle()
    assert not feed.has_next()


def test_mock_realtime_feed_saves_cursor_and_continues(tmp_path):
    csv_path = tmp_path / "candles.csv"
    cursor_path = tmp_path / "feed_cursor.json"
    write_candles(csv_path, rows=10)
    first_feed = MockRealtimeDataFeed(str(csv_path), window_size=3, max_steps=2, cursor_path=str(cursor_path))

    first_feed.connect()
    first_feed.get_next_candle()
    first_feed.get_next_candle()
    first_feed.disconnect()

    second_feed = MockRealtimeDataFeed(str(csv_path), window_size=3, max_steps=1, cursor_path=str(cursor_path))
    second_feed.connect()
    candle = second_feed.get_next_candle()

    assert cursor_path.exists()
    assert second_feed.start_index == 2
    assert str(candle["timestamp"]) == "2026-05-10 00:02:00+00:00"


def test_mock_realtime_feed_reset_cursor_restarts(tmp_path):
    csv_path = tmp_path / "candles.csv"
    cursor_path = tmp_path / "feed_cursor.json"
    write_candles(csv_path, rows=10)
    feed = MockRealtimeDataFeed(str(csv_path), window_size=3, max_steps=2, cursor_path=str(cursor_path))
    feed.get_next_candle()
    feed.get_next_candle()

    replay = MockRealtimeDataFeed(str(csv_path), window_size=3, cursor_path=str(cursor_path), reset_cursor=True)

    assert replay.start_index == 0
