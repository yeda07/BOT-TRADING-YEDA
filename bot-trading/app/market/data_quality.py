import pandas as pd

from app.market.candle_schema import STANDARD_CANDLE_COLUMNS, normalize_candles


def validate_candle(candle: dict | pd.Series) -> bool:
    valid, _ = validate_candles_df(pd.DataFrame([dict(candle)]))
    return valid


def validate_candles_df(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = [column for column in STANDARD_CANDLE_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return False, errors

    data = df.copy()
    try:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    except Exception:
        errors.append("Invalid timestamp values.")

    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[STANDARD_CANDLE_COLUMNS].isna().any().any():
        errors.append("Missing values or NaN found.")
    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("Invalid prices: OHLC values must be positive.")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        errors.append("Invalid prices: high is below open, close or low.")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        errors.append("Invalid prices: low is above open, close or high.")
    if data.duplicated(subset=["timestamp", "asset", "timeframe_seconds", "source"]).any():
        errors.append("Duplicated candles found.")
    if not data["timestamp"].is_monotonic_increasing:
        errors.append("Candles are not sorted by timestamp.")

    if not data.empty:
        timeframe = int(pd.to_numeric(data["timeframe_seconds"], errors="coerce").dropna().mode().iloc[0])
        gaps = data["timestamp"].sort_values().diff().dropna().dt.total_seconds()
        if not gaps.empty and (gaps > timeframe * 2).any():
            errors.append("Large time gaps found.")
        if not gaps.empty and (gaps <= 0).any():
            errors.append("Non-increasing timestamps found.")

    return not errors, errors


def clean_candles_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STANDARD_CANDLE_COLUMNS)
    asset = str(df["asset"].dropna().iloc[0]) if "asset" in df and not df["asset"].dropna().empty else "UNKNOWN"
    timeframe = int(df["timeframe_seconds"].dropna().iloc[0]) if "timeframe_seconds" in df and not df["timeframe_seconds"].dropna().empty else 60
    source = str(df["source"].dropna().iloc[0]) if "source" in df and not df["source"].dropna().empty else "unknown"
    data = normalize_candles(df, asset=asset, timeframe_seconds=timeframe, source=source)
    data = data.dropna(subset=STANDARD_CANDLE_COLUMNS)
    data = data.drop_duplicates(subset=["timestamp", "asset", "timeframe_seconds", "source"], keep="last")
    return data.sort_values("timestamp").reset_index(drop=True)
