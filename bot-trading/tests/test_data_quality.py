import pandas as pd

from app.market.candle_schema import normalize_candles
from app.market.data_quality import clean_candles_df, validate_candle, validate_candles_df


def valid_candles(rows: int = 3) -> pd.DataFrame:
    prices = [100 + index for index in range(rows)]
    return normalize_candles(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-05-10", periods=rows, freq="min", tz="UTC"),
                "open": prices,
                "high": [price + 1 for price in prices],
                "low": [price - 1 for price in prices],
                "close": prices,
                "volume": [10] * rows,
            }
        ),
        asset="EURUSD-OTC",
        timeframe_seconds=60,
        source="test",
    )


def test_validate_candle_rejects_negative_prices():
    candle = valid_candles(1).iloc[0].to_dict()
    candle["open"] = -1

    assert validate_candle(candle) is False


def test_validate_candles_df_detects_missing_columns():
    valid, errors = validate_candles_df(pd.DataFrame({"timestamp": ["2026-01-01"]}))

    assert valid is False
    assert any("Missing columns" in error for error in errors)


def test_clean_candles_df_removes_duplicates():
    candles = valid_candles(2)
    dirty = pd.concat([candles, candles.iloc[[0]]], ignore_index=True)

    cleaned = clean_candles_df(dirty)

    assert len(cleaned) == 2
