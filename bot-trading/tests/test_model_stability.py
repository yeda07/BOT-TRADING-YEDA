from app.validation.model_stability import ModelStabilityAnalyzer


def test_model_stability_analyzer_classifies_unstable_model():
    result = ModelStabilityAnalyzer().analyze(
        [
            {"win_rate": 0.9, "profit_factor": 3.0, "max_drawdown": 0.01},
            {"win_rate": 0.2, "profit_factor": 0.5, "max_drawdown": 0.4},
        ],
        [{"total_trades": 10}],
        [{"survived": False}, {"survived": False}],
    )

    assert result["status"] in {"UNSTABLE", "REJECTED"}
