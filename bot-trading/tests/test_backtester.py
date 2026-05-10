import pandas as pd
from unittest.mock import Mock

from app.execution.backtester import Backtester
from app.risk.risk_manager import RiskManager
from app.strategies.rule_based import RuleBasedStrategy
from app.strategies.rule_based import StrategySignal


def test_backtester_returns_metrics():
    rows = 120
    prices = [100 + i * 0.2 for i in range(rows)]
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.4 for p in prices],
            "low": [p - 0.4 for p in prices],
            "close": [p + 0.1 for p in prices],
            "volume": [100] * rows,
        }
    )
    tester = Backtester(RuleBasedStrategy(), RiskManager(min_model_confidence=0.5), initial_balance=1000)
    metrics, trades = tester.run(candles)
    assert metrics.final_balance > 0
    assert metrics.trades == len(trades)


class AlwaysBuyStrategy:
    def generate_signal(self, row: pd.Series) -> StrategySignal:
        return StrategySignal("BUY", 1.0, "Test buy.")


def test_backtester_logs_each_executed_trade():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=60, freq="min", tz="UTC"),
            "open": [100 + i for i in range(60)],
            "high": [101 + i for i in range(60)],
            "low": [99 + i for i in range(60)],
            "close": [100.5 + i for i in range(60)],
            "volume": [100] * 60,
        }
    )
    logger = Mock()
    tester = Backtester(
        AlwaysBuyStrategy(),
        RiskManager(min_model_confidence=0.0),
        initial_balance=1000,
        logger=logger,
        strategy_name="test_strategy",
    )
    metrics, trades = tester.run(candles)
    assert metrics.trades == len(trades)
    assert logger.info.call_count == len(trades)


def test_backtester_simulates_binary_options_and_saves_outputs(tmp_path):
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=240, freq="min", tz="UTC"),
            "open": [100 + i for i in range(240)],
            "high": [101 + i for i in range(240)],
            "low": [99 + i for i in range(240)],
            "close": [100.5 + i for i in range(240)],
            "volume": [100] * 240,
        }
    )
    tester = Backtester(
        AlwaysBuyStrategy(),
        RiskManager(min_model_confidence=0.0, min_candles=1),
        initial_balance=1000,
        payout=0.8,
        stake=10,
        expiration_candles=2,
        output_dir=tmp_path,
    )

    metrics, trades = tester.run(candles)

    assert metrics.initial_balance == 1000
    assert metrics.total_trades == len(trades)
    assert metrics.wins == len(trades)
    assert metrics.losses == 0
    assert metrics.win_rate == 1.0
    assert metrics.net_profit == len(trades) * 8
    assert metrics.final_balance == 1000 + metrics.net_profit
    assert metrics.max_consecutive_losses == 0
    assert len(metrics.equity_curve) == len(trades) + 1
    assert (tmp_path / "backtest_results.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()
