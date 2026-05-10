from datetime import datetime, timezone

import pandas as pd

from app.brokers.base import Broker, Order
from app.risk.risk_manager import RiskManager, RiskState


class Executor:
    def __init__(self, broker: Broker, strategy, risk_manager: RiskManager, state: RiskState) -> None:
        self.broker = broker
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.state = state

    def execute_latest(self, candles: pd.DataFrame, symbol: str) -> dict[str, str | float]:
        signal = self.strategy.generate_signal(candles.iloc[-1])
        decision = self.risk_manager.evaluate(signal.signal, signal.confidence, candles, self.state)
        if not decision.allowed:
            return {"status": "blocked", "reason": decision.reason}

        order = Order(
            symbol=symbol,
            side=signal.signal,
            amount=decision.stake,
            price=float(candles.iloc[-1]["close"]),
            timestamp=datetime.now(timezone.utc),
        )
        result = self.broker.place_order(order)
        return {
            "status": result.status,
            "message": result.message,
            "signal": signal.signal,
            "stake": decision.stake,
        }

