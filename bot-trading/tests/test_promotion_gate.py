import pytest

from app.ml.promotion_gate import DemoPromotionThresholds, assert_artifact_eligible_for_demo, evaluate_demo_promotion


def passing_test_metrics() -> dict:
    return {
        "win_rate_simulado": 0.58,
        "profit_factor_simulado": 1.35,
        "total_trades_simulado": 240,
        "roc_auc": 0.64,
        "max_drawdown_simulado": 0.06,
        "split": "test",
    }


def test_demo_promotion_gate_passes_only_on_unseen_strong_results():
    result = evaluate_demo_promotion(passing_test_metrics(), breakeven_win_rate=0.535)

    assert result["eligible"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("win_rate_simulado", 0.5417),
        ("profit_factor_simulado", 1.028),
        ("total_trades_simulado", 120),
        ("roc_auc", 0.559),
        ("max_drawdown_simulado", 0.18),
        ("split", "validation"),
    ],
)
def test_demo_promotion_gate_blocks_weak_or_unseen_failures(field, value):
    metrics = passing_test_metrics()
    metrics[field] = value

    result = evaluate_demo_promotion(metrics, breakeven_win_rate=0.535)

    assert result["eligible"] is False


def test_demo_promotion_gate_supports_stricter_trade_count():
    metrics = passing_test_metrics()
    thresholds = DemoPromotionThresholds(min_total_trades=300)

    result = evaluate_demo_promotion(metrics, breakeven_win_rate=0.535, thresholds=thresholds)

    assert result["eligible"] is False
    assert result["checks"]["total_trades"] is False


def test_assert_artifact_eligible_for_demo_blocks_unapproved_model():
    artifact = {"eligible_for_demo": False, "demo_gate": {"observed": passing_test_metrics()}}

    with pytest.raises(RuntimeError, match="blocked from demo mode"):
        assert_artifact_eligible_for_demo(artifact)
