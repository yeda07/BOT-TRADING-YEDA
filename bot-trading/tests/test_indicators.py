import pandas as pd
import pytest

from app.market.candles import load_candles_csv, validate_candles
from app.market.indicators import add_indicators, rsi


def sample_candles(rows: int = 80) -> pd.DataFrame:
    prices = [100 + i * 0.1 for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": [p + 0.05 for p in prices],
            "volume": [100] * rows,
        }
    )


def test_rsi_bounds():
    values = rsi(pd.Series([1, 2, 3, 2, 4, 5, 6, 7, 6, 8, 9, 10, 11, 10, 12, 13]), 5).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_add_indicators_columns():
    result = add_indicators(sample_candles())
    for column in ["sma_fast", "sma_slow", "rsi", "macd", "atr", "adx"]:
        assert column in result.columns


def test_candles_require_volume_column():
    candles = sample_candles().drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        validate_candles(candles)


def test_validate_candles_requires_datetime_timestamp():
    candles = sample_candles(200)
    candles["timestamp"] = candles["timestamp"].astype(str)
    with pytest.raises(ValueError, match="datetime"):
        validate_candles(candles)


def test_validate_candles_requires_chronological_order():
    candles = sample_candles(200)
    candles = candles.sort_values("timestamp", ascending=False).reset_index(drop=True)
    with pytest.raises(ValueError, match="ordered"):
        validate_candles(candles)


def test_validate_candles_requires_minimum_200_rows():
    candles = sample_candles(199)
    with pytest.raises(ValueError, match="200"):
        validate_candles(candles)


def test_load_candles_rejects_empty_prices(tmp_path):
    csv_path = tmp_path / "candles.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,100,101,99,100.5,1000\n"
        "2026-01-01T00:01:00Z,,102,100,101.5,1000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prices"):
        load_candles_csv(csv_path)


def test_load_candles_rejects_unsorted_timestamps(tmp_path):
    csv_path = tmp_path / "candles.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:01:00Z,101,102,100,101.5,1000\n"
        "2026-01-01T00:00:00Z,100,101,99,100.5,1000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordered"):
        load_candles_csv(csv_path)
