import pandas as pd
import pytest

from app.market.data_feed import CSVDataFeed


def write_candles(path, rows: int = 5) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-10", periods=rows, freq="min", tz="UTC")[::-1],
            "open": range(100, 100 + rows),
            "high": range(101, 101 + rows),
            "low": range(99, 99 + rows),
            "close": range(100, 100 + rows),
            "volume": [100] * rows,
        }
    )
    data.to_csv(path, index=False)
    return data


def test_csv_data_feed_loads_and_sorts_candles(tmp_path):
    csv_path = tmp_path / "candles.csv"
    write_candles(csv_path, rows=4)

    feed = CSVDataFeed(str(csv_path), window_size=2)
    first = feed.get_next_candle()

    assert first["timestamp"] == pd.Timestamp("2026-05-10 00:00:00+0000", tz="UTC")


def test_csv_data_feed_does_not_return_future_candles(tmp_path):
    csv_path = tmp_path / "candles.csv"
    write_candles(csv_path, rows=5)

    feed = CSVDataFeed(str(csv_path), window_size=3)
    feed.get_next_candle()
    latest = feed.get_latest_candles()

    assert len(latest) == 1
    assert latest["timestamp"].max() == pd.Timestamp("2026-05-10 00:00:00+0000", tz="UTC")


def test_csv_data_feed_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="CSV candles file not found"):
        CSVDataFeed(str(tmp_path / "missing.csv"))


def test_csv_data_feed_reports_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        CSVDataFeed(str(csv_path))
