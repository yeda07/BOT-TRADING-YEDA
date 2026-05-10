from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


def load_candles_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("CSV contains invalid timestamps.")

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("CSV does not contain usable candle rows.")
    return df


def validate_candles(df: pd.DataFrame, min_rows: int = 50) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Candles are missing required columns: {sorted(missing)}")
    if len(df) < min_rows:
        raise ValueError(f"Insufficient data: expected at least {min_rows} candles, got {len(df)}.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Candle prices must be positive.")
