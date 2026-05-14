import numpy as np
import pandas as pd

from app.market.indicators import add_indicators
from app.market.candles import REQUIRED_COLUMNS


FEATURE_COLUMNS = [
    "ema_9",
    "ema_21",
    "ema_diff",
    "ema_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_high",
    "bb_mid",
    "bb_low",
    "bb_width",
    "bb_position",
    "atr_14",
    "body",
    "candle_range",
    "upper_wick",
    "lower_wick",
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",
    "rolling_mean_5",
    "rolling_mean_10",
    "rolling_std_5",
    "rolling_std_10",
    "momentum_3",
    "momentum_5",
    "direction",
    "hour",
    "minute",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    _validate_ohlcv(df)
    featured = add_indicators(df)
    featured["ema_diff"] = featured["ema_9"] - featured["ema_21"]
    featured["ema_ratio"] = featured["ema_9"] / featured["ema_21"]
    featured["rsi_14"] = featured["rsi"]
    featured["bb_high"] = featured["bollinger_high"]
    featured["bb_mid"] = featured["bollinger_mid"]
    featured["bb_low"] = featured["bollinger_low"]
    featured["bb_width"] = featured["bb_high"] - featured["bb_low"]
    featured["bb_position"] = (featured["close"] - featured["bb_low"]) / featured["bb_width"]
    featured["atr_14"] = featured["atr"]
    featured["body"] = featured["candle_body"]
    featured["upper_wick"] = featured["high"] - featured[["open", "close"]].max(axis=1)
    featured["lower_wick"] = featured[["open", "close"]].min(axis=1) - featured["low"]
    featured["return_1"] = featured["close"].pct_change()
    featured["return_2"] = featured["close"].pct_change(2)
    featured["return_3"] = featured["close"].pct_change(3)
    featured["return_5"] = featured["close"].pct_change(5)
    featured["return_10"] = featured["close"].pct_change(10)
    featured["rolling_mean_5"] = featured["close"].rolling(window=5, min_periods=5).mean()
    featured["rolling_mean_10"] = featured["close"].rolling(window=10, min_periods=10).mean()
    featured["rolling_std_5"] = featured["close"].rolling(window=5, min_periods=5).std()
    featured["rolling_std_10"] = featured["close"].rolling(window=10, min_periods=10).std()
    featured["momentum_3"] = featured["close"] - featured["close"].shift(3)
    featured["momentum_5"] = featured["close"] - featured["close"].shift(5)
    featured["hour"] = featured["timestamp"].dt.hour
    featured["minute"] = featured["timestamp"].dt.minute

    # Backward-compatible feature aliases used by the current rule strategy/risk manager.
    featured["sma_ratio"] = featured["sma_fast"] / featured["sma_slow"] - 1
    featured["atr_ratio"] = featured["atr"] / featured["close"]
    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured.dropna().reset_index(drop=True)


def create_features(
    df: pd.DataFrame,
    expiration_candles: int = 1,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if expiration_candles < 1:
        raise ValueError("expiration_candles must be at least 1.")

    featured = build_features(df)
    if len(featured) <= expiration_candles:
        raise ValueError("Insufficient rows after feature engineering.")

    dataset = featured.copy()
    dataset["future_close"] = dataset["close"].shift(-expiration_candles)
    dataset = dataset.dropna(subset=["future_close"])
    dataset["target"] = np.where(dataset["future_close"] > dataset["close"], 1, 0)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["target"])

    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset["target"].astype(int).copy()
    if X.empty or y.empty:
        raise ValueError("Feature engineering produced an empty supervised dataset.")
    if len(X) != len(y):
        raise ValueError("Feature matrix and target size mismatch.")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Feature matrix or target contains NaN values.")
    return X.reset_index(drop=True), y.reset_index(drop=True), FEATURE_COLUMNS.copy()


def build_supervised_dataset(
    df: pd.DataFrame,
    expiration_candles: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    X, y, _ = create_features(df, expiration_candles=expiration_candles)
    return X, y


def _validate_ohlcv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")
    if len(df) < 60:
        raise ValueError("At least 60 candles are required to create features.")
