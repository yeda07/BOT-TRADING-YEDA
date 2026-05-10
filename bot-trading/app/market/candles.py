from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_candles_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    _validate_required_columns(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("CSV contains invalid timestamps.")

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("CSV contains empty or invalid prices.")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    if df.empty:
        raise ValueError("CSV does not contain usable candle rows.")
    _validate_timestamp_dtype(df)
    _validate_chronological_order(df)
    return df


def validate_candles(df: pd.DataFrame, min_rows: int = 200) -> None:
    _validate_required_columns(df)
    _validate_timestamp_dtype(df)
    _validate_chronological_order(df)
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Candles contain empty prices.")
    if len(df) < min_rows:
        raise ValueError(f"Insufficient data: expected at least {min_rows} candles, got {len(df)}.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Candle prices must be positive.")


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Candles are missing required columns: {sorted(missing)}")


def _validate_timestamp_dtype(df: pd.DataFrame) -> None:
    if not is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("Timestamp column must be datetime.")
    if df["timestamp"].isna().any():
        raise ValueError("Candles contain invalid timestamps.")


def _validate_chronological_order(df: pd.DataFrame) -> None:
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Candles must be ordered by timestamp.")
