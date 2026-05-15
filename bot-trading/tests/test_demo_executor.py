from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from app.brokers.demo_stub_broker import DemoStubBroker
from app.execution.demo_executor import DemoExecutor
from app.execution.execution_guard import ExecutionGuard
from app.execution.trade_logger import TradeLogger
from app.risk.risk_manager import RiskManager
from app.storage.trades_repository import TradesRepository


def cfg(tmp_path):
    return SimpleNamespace(
        BOT_MODE="demo",
        ENABLE_REAL_TRADING=False,
        MIN_CONFIDENCE=0.58,
        PAYOUT=0.87,
        MIN_PROFIT_FACTOR=1.0,
        MAX_ALLOWED_DRAWDOWN=0.15,
        KILL_SWITCH_PATH=str(tmp_path / "kill.json"),
        EXECUTION_STATE_PATH=str(tmp_path / "state.json"),
        EXPIRATION_CANDLES=1,
        TIMEFRAME_SECONDS=60,
    )


def model(path):
    joblib.dump(
        {
            "breakeven_win_rate": 0.5,
            "metrics": {"test": {"win_rate_simulado": 0.7, "profit_factor_simulado": 1.3, "max_drawdown_simulado": 0.05}},
        },
        path,
    )


def test_demo_stub_broker_returns_demo_account_type():
    assert DemoStubBroker().get_account_type() == "demo"


def test_demo_executor_blocks_real_account_type(tmp_path):
    class RealDemoStub(DemoStubBroker):
        def get_account_type(self):
            return "real"

    path = tmp_path / "model.joblib"
    model(path)
    settings = cfg(tmp_path)
    broker = RealDemoStub()
    broker.connect()
    risk = RiskManager(balance=1000)
    executor = DemoExecutor(
        broker,
        ExecutionGuard(settings, risk, str(path)),
        risk,
        TradeLogger(str(tmp_path / "trades.csv")),
        TradesRepository(str(tmp_path / "trades.db")),
        settings,
    )

    with pytest.raises(RuntimeError, match="not demo"):
        executor.execute("BUY", 0.9, "EURUSD-OTC", pd.Series({"timestamp": "t", "close": 100}))


def test_demo_executor_resolves_order(tmp_path):
    path = tmp_path / "model.joblib"
    model(path)
    settings = cfg(tmp_path)
    broker = DemoStubBroker(initial_balance=1000)
    broker.connect()
    risk = RiskManager(balance=1000)
    repo = TradesRepository(str(tmp_path / "trades.db"))
    executor = DemoExecutor(
        broker,
        ExecutionGuard(settings, risk, str(path)),
        risk,
        TradeLogger(str(tmp_path / "trades.csv")),
        repo,
        settings,
    )

    result = executor.execute("BUY", 0.9, "EURUSD-OTC", pd.Series({"timestamp": "t", "close": 100}))

    assert result["status"] == "RESOLVED"
    assert repo.get_summary()["total_trades"] == 1
