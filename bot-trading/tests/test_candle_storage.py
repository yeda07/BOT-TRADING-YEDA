import pandas as pd

from app.market.candle_schema import normalize_candles
from app.market.candle_storage import CandleStorage


def candles():
    return normalize_candles(
        pd.DataFrame(
            {
                "timestamp": ["2026-05-10T00:00:00Z"],
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [1],
            }
        ),
        asset="EURUSD-OTC",
        timeframe_seconds=60,
        source="test",
    )


def test_candle_storage_creates_file(tmp_path):
    path = tmp_path / "collected.csv"
    storage = CandleStorage(str(path))

    assert path.exists()
    assert storage.read_all().empty


def test_candle_storage_does_not_duplicate_candles(tmp_path):
    storage = CandleStorage(str(tmp_path / "collected.csv"))

    storage.append_candles(candles())
    storage.append_candles(candles())

    assert len(storage.read_all()) == 1
