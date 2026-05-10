import pandas as pd

from app.risk.risk_manager import RiskManager, RiskState


def candles(rows: int = 60, adx: float = 25.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0] * rows,
            "adx": [adx] * rows,
            "atr_ratio": [0.001] * rows,
        }
    )


def test_blocks_realistic_low_confidence():
    manager = RiskManager(min_model_confidence=0.6)
    decision = manager.evaluate("BUY", 0.4, candles(), RiskState(1000, 1000))
    assert not decision.allowed


def test_blocks_three_consecutive_losses():
    manager = RiskManager(max_consecutive_losses=3)
    state = RiskState(950, 1000, consecutive_losses=3)
    decision = manager.evaluate("BUY", 0.8, candles(), state)
    assert not decision.allowed


def test_allows_valid_trade_and_sizes_stake():
    manager = RiskManager(risk_per_trade=0.01)
    decision = manager.evaluate("BUY", 0.8, candles(), RiskState(1000, 1000))
    assert decision.allowed
    assert decision.stake == 10.0

