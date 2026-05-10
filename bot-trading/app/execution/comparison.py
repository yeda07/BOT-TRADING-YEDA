from dataclasses import asdict, dataclass
from logging import Logger
from pathlib import Path
from typing import Callable

import pandas as pd

from app.execution.backtester import BacktestMetrics, Backtester
from app.risk.risk_manager import RiskManager
from app.strategies.ml_strategy import MLStrategy
from app.strategies.rule_based import RuleBasedStrategy


@dataclass(frozen=True)
class StrategyComparison:
    rule_based: BacktestMetrics
    ml: BacktestMetrics
    winner: str
    winner_metric: str

    def to_dict(self) -> dict:
        return {
            "rule_based": asdict(self.rule_based),
            "ml": asdict(self.ml),
            "winner": self.winner,
            "winner_metric": self.winner_metric,
        }


def compare_rule_vs_ml(
    candles: pd.DataFrame,
    model_path: str | Path,
    risk_manager_factory: Callable[[], RiskManager],
    initial_balance: float = 1000.0,
    payout: float = 0.80,
    logger: Logger | None = None,
    winner_metric: str = "net_profit",
) -> tuple[StrategyComparison, pd.DataFrame]:
    rule_metrics, rule_trades = Backtester(
        strategy=RuleBasedStrategy(),
        risk_manager=risk_manager_factory(),
        initial_balance=initial_balance,
        payout=payout,
        logger=logger,
        strategy_name="rule_based",
    ).run(candles)

    ml_metrics, ml_trades = Backtester(
        strategy=MLStrategy(model_path),
        risk_manager=risk_manager_factory(),
        initial_balance=initial_balance,
        payout=payout,
        logger=logger,
        strategy_name="ml",
    ).run(candles)

    winner = _winner(rule_metrics, ml_metrics, winner_metric)
    comparison = StrategyComparison(
        rule_based=rule_metrics,
        ml=ml_metrics,
        winner=winner,
        winner_metric=winner_metric,
    )
    trades = pd.concat([rule_trades, ml_trades], ignore_index=True)
    return comparison, trades


def _winner(rule_metrics: BacktestMetrics, ml_metrics: BacktestMetrics, metric: str) -> str:
    if not hasattr(rule_metrics, metric) or not hasattr(ml_metrics, metric):
        raise ValueError(f"Unsupported winner metric: {metric}")

    rule_value = getattr(rule_metrics, metric)
    ml_value = getattr(ml_metrics, metric)
    if rule_value == ml_value:
        return "tie"
    return "rule_based" if rule_value > ml_value else "ml"
