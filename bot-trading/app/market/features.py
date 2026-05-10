import numpy as np
import pandas as pd

from app.market.indicators import add_indicators


FEATURE_COLUMNS = [
    "ema_9",
    "ema_21",
    "ema_diff",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_high",
    "bb_mid",
    "bb_low",
    "bb_position",
    "atr_14",
    "body",
    "candle_range",
    "return_1",
    "return_3",
    "return_5",
    "direction",
    "hour",
    "minute",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_indicators(df)
    featured["ema_diff"] = featured["ema_9"] - featured["ema_21"]
    featured["rsi_14"] = featured["rsi"]
    featured["bb_high"] = featured["bollinger_high"]
    featured["bb_mid"] = featured["bollinger_mid"]
    featured["bb_low"] = featured["bollinger_low"]
    featured["bb_position"] = (featured["close"] - featured["bb_low"]) / (featured["bb_high"] - featured["bb_low"])
    featured["atr_14"] = featured["atr"]
    featured["body"] = featured["candle_body"]
    featured["return_1"] = featured["close"].pct_change()
    featured["return_3"] = featured["close"].pct_change(3)
    featured["return_5"] = featured["close"].pct_change(5)
    featured["hour"] = featured["timestamp"].dt.hour
    featured["minute"] = featured["timestamp"].dt.minute

    # Backward-compatible feature aliases used by the current rule strategy/risk manager.
    featured["sma_ratio"] = featured["sma_fast"] / featured["sma_slow"] - 1
    featured["ema_ratio"] = featured["ema_fast"] / featured["ema_slow"] - 1
    featured["atr_ratio"] = featured["atr"] / featured["close"]
    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured.dropna().reset_index(drop=True)


def build_supervised_dataset(
    df: pd.DataFrame,
    expiration_candles: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    if expiration_candles < 1:
        raise ValueError("expiration_candles must be at least 1.")

    featured = build_features(df)
    dataset = featured[FEATURE_COLUMNS].copy()
    dataset["future_close"] = featured["close"].shift(-expiration_candles)
    dataset["target"] = np.where(dataset["future_close"] > featured["close"], 1, 0)
    dataset = dataset.dropna()
    return dataset[FEATURE_COLUMNS], dataset["target"].astype(int)
