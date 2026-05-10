from dataclasses import dataclass
from typing import Literal

import pandas as pd


Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class StrategySignal:
    signal: Signal
    confidence: float
    reason: str


class RuleBasedStrategy:
    def generate_signal(self, row: pd.Series) -> StrategySignal:
        required = ["sma_fast", "sma_slow", "rsi", "macd", "macd_signal", "adx"]
        if row[required].isna().any():
            return StrategySignal("HOLD", 0.0, "Insufficient indicators.")

        trend_up = row["sma_fast"] > row["sma_slow"] and row["macd"] > row["macd_signal"]
        trend_down = row["sma_fast"] < row["sma_slow"] and row["macd"] < row["macd_signal"]

        if row["adx"] < 18:
            return StrategySignal("HOLD", 0.2, "Lateral market.")
        if trend_up and 45 <= row["rsi"] <= 70:
            return StrategySignal("BUY", 0.68, "Trend and momentum aligned bullish.")
        if trend_down and 30 <= row["rsi"] <= 55:
            return StrategySignal("SELL", 0.68, "Trend and momentum aligned bearish.")
        return StrategySignal("HOLD", 0.35, "No clear setup.")

