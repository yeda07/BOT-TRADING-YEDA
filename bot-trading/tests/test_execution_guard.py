from types import SimpleNamespace

import joblib

from app.execution.execution_guard import ExecutionGuard
from app.execution.kill_switch import KillSwitch
from app.risk.risk_manager import RiskManager


def settings(tmp_path):
    return SimpleNamespace(
        BOT_MODE="demo",
        ENABLE_REAL_TRADING=False,
        MIN_CONFIDENCE=0.58,
        PAYOUT=0.87,
        MIN_PROFIT_FACTOR=1.0,
        MAX_ALLOWED_DRAWDOWN=0.15,
        KILL_SWITCH_PATH=str(tmp_path / "kill.json"),
        EXECUTION_STATE_PATH=str(tmp_path / "state.json"),
    )


def write_artifact(path, win_rate=0.7, profit_factor=1.2):
    joblib.dump(
        {
            "breakeven_win_rate": 0.534,
            "metrics": {"test": {"win_rate_simulado": win_rate, "profit_factor_simulado": profit_factor, "max_drawdown_simulado": 0.05}},
        },
        path,
    )


def test_execution_guard_blocks_hold(tmp_path):
    model = tmp_path / "model.joblib"
    write_artifact(model)
    guard = ExecutionGuard(settings(tmp_path), RiskManager(), str(model))

    allowed, reason = guard.validate_before_order("HOLD", 0.9, 1000, "EURUSD-OTC", "t")

    assert not allowed
    assert "HOLD" in reason


def test_execution_guard_blocks_low_confidence(tmp_path):
    model = tmp_path / "model.joblib"
    write_artifact(model)
    guard = ExecutionGuard(settings(tmp_path), RiskManager(), str(model))

    allowed, _ = guard.validate_before_order("BUY", 0.1, 1000, "EURUSD-OTC", "t")

    assert not allowed


def test_execution_guard_blocks_active_kill_switch(tmp_path):
    model = tmp_path / "model.joblib"
    cfg = settings(tmp_path)
    write_artifact(model)
    KillSwitch(cfg.KILL_SWITCH_PATH).activate("manual")
    guard = ExecutionGuard(cfg, RiskManager(), str(model))

    allowed, reason = guard.validate_before_order("BUY", 0.9, 1000, "EURUSD-OTC", "t")

    assert not allowed
    assert "Kill switch" in reason


def test_execution_guard_blocks_model_below_breakeven(tmp_path):
    model = tmp_path / "model.joblib"
    write_artifact(model, win_rate=0.4)
    guard = ExecutionGuard(settings(tmp_path), RiskManager(), str(model))

    allowed, reason = guard.validate_before_order("BUY", 0.9, 1000, "EURUSD-OTC", "t")

    assert not allowed
    assert "breakeven" in reason
