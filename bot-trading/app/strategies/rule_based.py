from dataclasses import dataclass
from typing import Literal

import pandas as pd


Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class StrategySignal:
    signal: Signal
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Signal | float | str]:
        return {"signal": self.signal, "confidence": self.confidence, "reason": self.reason}


class RuleBasedStrategy:
    def __init__(self, min_atr: float = 0.000001, lateral_market_adx_threshold: float = 18.0) -> None:
        self.min_atr = min_atr
        self.lateral_market_adx_threshold = lateral_market_adx_threshold

    def generate_signal(self, row: pd.Series) -> StrategySignal:
        required = ["ema_9", "ema_21", "rsi", "close", "macd_diff", "atr"]
        if any(column not in row.index for column in required) or row[required].isna().any():
            return StrategySignal("HOLD", 0.0, "Insufficient data.")

        if row["rsi"] > 70 or row["rsi"] < 30:
            return StrategySignal("HOLD", 0.2, "RSI is extreme.")
        if row["atr"] <= self.min_atr:
            return StrategySignal("HOLD", 0.2, "Low volatility.")
        if "adx" in row.index and pd.notna(row["adx"]) and row["adx"] < self.lateral_market_adx_threshold:
            return StrategySignal("HOLD", 0.25, "Market is lateral.")

        buy_conditions = [
            row["ema_9"] > row["ema_21"],
            50 <= row["rsi"] <= 70,
            row["close"] > row["ema_21"],
            row["macd_diff"] > 0,
            row["atr"] > self.min_atr,
        ]
        sell_conditions = [
            row["ema_9"] < row["ema_21"],
            30 <= row["rsi"] <= 50,
            row["close"] < row["ema_21"],
            row["macd_diff"] < 0,
            row["atr"] > self.min_atr,
        ]

        if all(buy_conditions):
            return StrategySignal("BUY", 0.72, "EMA, RSI, price, MACD and ATR aligned bullish.")
        if all(sell_conditions):
            return StrategySignal("SELL", 0.72, "EMA, RSI, price, MACD and ATR aligned bearish.")
        return StrategySignal("HOLD", 0.35, "Mixed signals.")
