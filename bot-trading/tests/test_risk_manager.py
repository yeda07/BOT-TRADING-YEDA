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


def test_default_position_size_never_risks_more_than_one_percent():
    manager = RiskManager(balance=3000)
    assert manager.can_trade()
    assert manager.get_position_size() == 30.0


def test_position_size_caps_requested_risk_at_one_percent():
    manager = RiskManager(balance=3000, risk_per_trade=0.02)
    assert manager.get_position_size() == 30.0


def test_register_result_updates_balance_and_blocks_daily_loss():
    manager = RiskManager(balance=1000, max_daily_loss_pct=0.05)
    manager.register_result(-60)
    assert manager.balance == 940
    assert manager.daily_pnl == -60
    assert not manager.can_trade()
    assert manager.get_position_size() == 0.0


def test_register_result_blocks_after_max_consecutive_losses():
    manager = RiskManager(balance=1000, max_consecutive_losses=2)
    manager.register_result(-10)
    manager.register_result(-10)
    assert manager.consecutive_losses == 2
    assert not manager.can_trade()


def test_register_result_resets_consecutive_losses_after_win():
    manager = RiskManager(balance=1000)
    manager.register_result(-10)
    manager.register_result(5)
    assert manager.consecutive_losses == 0
    assert manager.balance == 995


def test_reset_daily_limits_clears_daily_blocks():
    manager = RiskManager(balance=1000, max_daily_loss_pct=0.05)
    manager.register_result(-60)
    assert not manager.can_trade()
    manager.reset_daily_limits()
    assert manager.daily_pnl == 0.0
    assert manager.consecutive_losses == 0
    assert manager.initial_balance == manager.balance
    assert manager.can_trade()
