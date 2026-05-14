from dataclasses import asdict, dataclass
from logging import Logger
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd

from app.execution.backtester import BacktestMetrics, Backtester
from app.execution.ml_backtester import MLBacktestMetrics, MLBacktester
from app.market.features import FEATURE_COLUMNS
from app.risk.risk_manager import RiskManager
from app.strategies.rule_based import RuleBasedStrategy


@dataclass(frozen=True)
class StrategyComparison:
    rule_based: BacktestMetrics
    ml: MLBacktestMetrics
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
    expiration_candles: int = 1,
    logger: Logger | None = None,
    winner_metric: str = "net_profit",
    output_path: str | Path = "data/logs/strategy_comparison.csv",
) -> tuple[StrategyComparison, pd.DataFrame]:
    rule_metrics, rule_trades = Backtester(
        strategy=RuleBasedStrategy(),
        risk_manager=risk_manager_factory(),
        initial_balance=initial_balance,
        payout=payout,
        expiration_candles=expiration_candles,
        logger=logger,
        strategy_name="rule_based",
    ).run(candles)

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    features = artifact.get("features", FEATURE_COLUMNS) if isinstance(artifact, dict) else FEATURE_COLUMNS
    min_confidence = artifact.get("min_confidence", 0.58) if isinstance(artifact, dict) else 0.58
    ml_metrics, ml_trades = MLBacktester(
        model=model,
        features=features,
        stake=max(1.0, initial_balance * risk_manager_factory().risk_per_trade),
        initial_balance=initial_balance,
        payout=payout,
        expiration_candles=expiration_candles,
        min_confidence=min_confidence,
    ).run(candles)
    ml_trades = ml_trades.copy()
    if not ml_trades.empty:
        ml_trades.insert(1, "strategy", "ml")

    winner = _winner(rule_metrics, ml_metrics, winner_metric)
    comparison = StrategyComparison(
        rule_based=rule_metrics,
        ml=ml_metrics,
        winner=winner,
        winner_metric=winner_metric,
    )
    trades = pd.concat([rule_trades, ml_trades], ignore_index=True)
    save_strategy_comparison(rule_metrics, ml_metrics, output_path=output_path)
    if logger:
        logger.info("Strategy comparison saved to %s", output_path)
    return comparison, trades


def save_strategy_comparison(
    rule_metrics: BacktestMetrics,
    ml_metrics: MLBacktestMetrics,
    output_path: str | Path = "data/logs/strategy_comparison.csv",
) -> pd.DataFrame:
    rows = [
        _comparison_row("rule_based", rule_metrics),
        _comparison_row("ml", ml_metrics),
    ]
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def format_strategy_comparison(rule_metrics: BacktestMetrics, ml_metrics: MLBacktestMetrics) -> str:
    return (
        "===== STRATEGY COMPARISON =====\n"
        "Rule Based:\n"
        f"Win Rate: {rule_metrics.win_rate:.2%}\n"
        f"Profit: {rule_metrics.net_profit:.2f}\n"
        f"Max Drawdown: {rule_metrics.max_drawdown:.2%}\n\n"
        "ML Strategy:\n"
        f"Win Rate: {ml_metrics.win_rate:.2%}\n"
        f"Profit: {ml_metrics.net_profit:.2f}\n"
        f"Max Drawdown: {ml_metrics.max_drawdown:.2%}"
    )


def _comparison_row(strategy: str, metrics: BacktestMetrics | MLBacktestMetrics) -> dict:
    return {
        "strategy": strategy,
        "total_trades": metrics.total_trades,
        "wins": metrics.wins,
        "losses": metrics.losses,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "max_drawdown": metrics.max_drawdown,
        "net_profit": metrics.net_profit,
        "final_balance": metrics.final_balance,
    }


def _winner(rule_metrics: BacktestMetrics, ml_metrics: MLBacktestMetrics, metric: str) -> str:
    if not hasattr(rule_metrics, metric) or not hasattr(ml_metrics, metric):
        raise ValueError(f"Unsupported winner metric: {metric}")

    rule_value = getattr(rule_metrics, metric)
    ml_value = getattr(ml_metrics, metric)
    if rule_value == ml_value:
        return "tie"
    return "rule_based" if rule_value > ml_value else "ml"
