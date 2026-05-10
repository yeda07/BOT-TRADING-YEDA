import pandas as pd
import pytest

from app.market.candles import validate_candles
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
