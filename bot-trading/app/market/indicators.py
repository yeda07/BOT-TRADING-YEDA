import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_value = atr(df, window).replace(0, np.nan)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / atr_value
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / atr_value
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(window=window, min_periods=window).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["sma_fast"] = sma(enriched["close"], 10)
    enriched["sma_slow"] = sma(enriched["close"], 30)
    enriched["ema_fast"] = ema(enriched["close"], 12)
    enriched["ema_slow"] = ema(enriched["close"], 26)
    enriched["rsi"] = rsi(enriched["close"], 14)
    enriched["atr"] = atr(enriched, 14)
    enriched["adx"] = adx(enriched, 14)
    return pd.concat([enriched, macd(enriched["close"])], axis=1)

