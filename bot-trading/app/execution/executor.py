import pandas as pd

from app.brokers.base import BrokerBase
from app.risk.risk_manager import RiskManager, RiskState


class Executor:
    def __init__(self, broker: BrokerBase, strategy, risk_manager: RiskManager, state: RiskState) -> None:
        self.broker = broker
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.state = state

    def execute_latest(self, candles: pd.DataFrame, symbol: str, expiration: int = 1) -> dict[str, str | float]:
        signal = self.strategy.generate_signal(candles.iloc[-1])
        decision = self.risk_manager.evaluate(signal.signal, signal.confidence, candles, self.state)
        if not decision.allowed:
            return {"status": "blocked", "reason": decision.reason}

        result = self.broker.place_order(
            asset=symbol,
            amount=decision.stake,
            direction=signal.signal,
            expiration=expiration,
        )
        return {
            "status": result.status,
            "message": result.message,
            "signal": signal.signal,
            "stake": decision.stake,
        }
