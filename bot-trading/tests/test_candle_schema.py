import pandas as pd
import pytest

from app.market.candle_schema import STANDARD_CANDLE_COLUMNS, normalize_candles


def test_candle_schema_adds_standard_columns():
    candles = pd.DataFrame(
        {
            "timestamp": ["2026-05-10T00:00:00Z"],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
        }
    )

    result = normalize_candles(candles, asset="EURUSD-OTC", timeframe_seconds=60, source="test")

    assert result.columns.tolist() == STANDARD_CANDLE_COLUMNS
    assert result.iloc[0]["volume"] == 0.0
    assert result.iloc[0]["asset"] == "EURUSD-OTC"


def test_candle_schema_rejects_invalid_high_low():
    candles = pd.DataFrame(
        {
            "timestamp": ["2026-05-10T00:00:00Z"],
            "open": [100],
            "high": [99],
            "low": [98],
            "close": [100],
            "volume": [1],
        }
    )

    with pytest.raises(ValueError, match="high"):
        normalize_candles(candles, asset="EURUSD-OTC", timeframe_seconds=60, source="test")
