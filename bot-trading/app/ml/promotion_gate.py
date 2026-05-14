from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DemoPromotionThresholds:
    min_edge_over_breakeven: float = 0.03
    min_profit_factor: float = 1.20
    min_total_trades: int = 200
    min_roc_auc: float = 0.60
    max_drawdown: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_DEMO_PROMOTION_THRESHOLDS = DemoPromotionThresholds()


def evaluate_demo_promotion(
    metrics: dict,
    breakeven_win_rate: float,
    thresholds: DemoPromotionThresholds = DEFAULT_DEMO_PROMOTION_THRESHOLDS,
) -> dict:
    """Return a deterministic gate result for promoting an ML model to demo mode."""
    win_rate = _metric(metrics, "win_rate_simulado")
    profit_factor = _metric(metrics, "profit_factor_simulado")
    total_trades = int(_metric(metrics, "total_trades_simulado", "total_samples"))
    roc_auc = _metric(metrics, "roc_auc")
    max_drawdown = _metric(metrics, "max_drawdown_simulado", default=1.0)
    required_win_rate = breakeven_win_rate + thresholds.min_edge_over_breakeven

    checks = {
        "win_rate_edge": win_rate >= required_win_rate,
        "profit_factor": profit_factor >= thresholds.min_profit_factor,
        "total_trades": total_trades >= thresholds.min_total_trades,
        "roc_auc": roc_auc >= thresholds.min_roc_auc,
        "max_drawdown": max_drawdown <= thresholds.max_drawdown,
        "unseen_test_data": metrics.get("split") == "test" or bool(metrics.get("is_test_split", False)),
    }

    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds.to_dict(),
        "observed": {
            "win_rate": win_rate,
            "required_win_rate": required_win_rate,
            "edge_over_breakeven": win_rate - breakeven_win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "roc_auc": roc_auc,
            "max_drawdown": max_drawdown,
            "split": metrics.get("split"),
        },
    }


def assert_artifact_eligible_for_demo(artifact: dict) -> None:
    if not isinstance(artifact, dict):
        raise RuntimeError("Demo mode requires a validated model artifact with demo gate metrics.")
    if not artifact.get("eligible_for_demo", False):
        gate = artifact.get("demo_gate", {})
        observed = gate.get("observed", {})
        raise RuntimeError(
            "ML model is blocked from demo mode. "
            f"Observed test metrics: win_rate={observed.get('win_rate', 0.0):.2%}, "
            f"profit_factor={observed.get('profit_factor', 0.0):.3f}, "
            f"roc_auc={observed.get('roc_auc', 0.0):.3f}, "
            f"total_trades={observed.get('total_trades', 0)}, "
            f"max_drawdown={observed.get('max_drawdown', 0.0):.2%}."
        )


def _metric(metrics: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        return value
    return default
