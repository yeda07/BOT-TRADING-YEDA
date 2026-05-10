import numpy as np
import pandas as pd

from app.market.indicators import add_indicators


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "sma_ratio",
    "ema_ratio",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_ratio",
    "adx",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_indicators(df)
    featured["return_1"] = featured["close"].pct_change()
    featured["return_3"] = featured["close"].pct_change(3)
    featured["sma_ratio"] = featured["sma_fast"] / featured["sma_slow"] - 1
    featured["ema_ratio"] = featured["ema_fast"] / featured["ema_slow"] - 1
    featured["atr_ratio"] = featured["atr"] / featured["close"]
    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured


def build_supervised_dataset(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    featured = build_features(df)
    future_return = featured["close"].shift(-horizon) / featured["close"] - 1
    target = np.where(future_return > threshold, 1, np.where(future_return < -threshold, -1, 0))
    dataset = featured[FEATURE_COLUMNS].copy()
    dataset["target"] = target
    dataset = dataset.dropna()
    return dataset[FEATURE_COLUMNS], dataset["target"].astype(int)

