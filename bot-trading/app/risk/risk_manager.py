from dataclasses import dataclass

import pandas as pd


@dataclass
class RiskState:
    balance: float
    starting_balance: float
    consecutive_losses: int = 0
    daily_pnl: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    stake: float
    reason: str


class RiskManager:
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_consecutive_losses: int = 3,
        max_daily_loss_pct: float = 0.05,
        min_model_confidence: float = 0.60,
        min_candles: int = 50,
        max_volatility_multiplier: float = 3.0,
        lateral_market_adx_threshold: float = 18.0,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss_pct = max_daily_loss_pct
        self.min_model_confidence = min_model_confidence
        self.min_candles = min_candles
        self.max_volatility_multiplier = max_volatility_multiplier
        self.lateral_market_adx_threshold = lateral_market_adx_threshold

    def evaluate(
        self,
        signal: str,
        confidence: float,
        candles: pd.DataFrame,
        state: RiskState,
    ) -> RiskDecision:
        if signal == "HOLD":
            return RiskDecision(False, 0.0, "Signal is HOLD.")
        if confidence < self.min_model_confidence:
            return RiskDecision(False, 0.0, "Model or strategy confidence is too low.")
        if len(candles) < self.min_candles:
            return RiskDecision(False, 0.0, "Insufficient market data.")
        if state.consecutive_losses >= self.max_consecutive_losses:
            return RiskDecision(False, 0.0, "Maximum consecutive losses reached.")
        if state.daily_pnl <= -(state.starting_balance * self.max_daily_loss_pct):
            return RiskDecision(False, 0.0, "Maximum daily loss reached.")

        latest = candles.iloc[-1]
        if pd.notna(latest.get("adx")) and latest["adx"] < self.lateral_market_adx_threshold:
            return RiskDecision(False, 0.0, "Market is lateral.")

        if "atr_ratio" in candles.columns:
            recent_atr = candles["atr_ratio"].dropna().tail(30)
            if len(recent_atr) >= 10 and latest["atr_ratio"] > recent_atr.median() * self.max_volatility_multiplier:
                return RiskDecision(False, 0.0, "Abnormal volatility.")

        stake = max(0.0, state.balance * self.risk_per_trade)
        return RiskDecision(True, stake, "Risk accepted.")

    @staticmethod
    def update_after_trade(state: RiskState, pnl: float) -> RiskState:
        state.balance += pnl
        state.daily_pnl += pnl
        state.consecutive_losses = state.consecutive_losses + 1 if pnl < 0 else 0
        return state

