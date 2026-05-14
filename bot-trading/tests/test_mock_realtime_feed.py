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
    feed = MockRealtimeDataFeed(str(csv_path), window_size=3, max_steps=2)

    feed.connect()
    first = feed.get_next_candle()
    latest = feed.get_latest_candles(10)

    assert len(latest) == 1
    assert latest.iloc[-1]["timestamp"] == first["timestamp"]
    assert feed.has_next()
    feed.get_next_candle()
    assert not feed.has_next()
