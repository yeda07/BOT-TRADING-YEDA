import pandas as pd


STANDARD_CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "asset",
    "timeframe_seconds",
    "source",
]


def normalize_candles(
    candles: pd.DataFrame,
    asset: str,
    timeframe_seconds: int,
    source: str,
) -> pd.DataFrame:
    data = candles.copy()
    if "volume" not in data.columns:
        data["volume"] = 0
    if "asset" not in data.columns:
        data["asset"] = asset
    if "timeframe_seconds" not in data.columns:
        data["timeframe_seconds"] = timeframe_seconds
    if "source" not in data.columns:
        data["source"] = source

    missing = [column for column in STANDARD_CANDLE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Candles are missing standard columns: {missing}")

    data = data[STANDARD_CANDLE_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
    data["timeframe_seconds"] = pd.to_numeric(data["timeframe_seconds"], errors="raise").astype(int)
    data["asset"] = data["asset"].astype(str)
    data["source"] = data["source"].astype(str)
    _validate_prices(data)
    return data.sort_values("timestamp").reset_index(drop=True)


def normalize_candle(candle: dict | pd.Series, asset: str, timeframe_seconds: int, source: str) -> pd.Series:
    return normalize_candles(pd.DataFrame([dict(candle)]), asset, timeframe_seconds, source).iloc[0]


def _validate_prices(data: pd.DataFrame) -> None:
    price_columns = ["open", "high", "low", "close"]
    if (data[price_columns] <= 0).any().any():
        raise ValueError("Candle prices must be positive.")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Candle high must be greater than or equal to open, close and low.")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Candle low must be lower than or equal to open, close and high.")
