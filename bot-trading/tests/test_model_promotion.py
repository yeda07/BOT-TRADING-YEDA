from types import SimpleNamespace

import joblib
import json

from app.mlops.model_promotion import ModelPromotionManager
from app.mlops.model_registry import ModelRegistry


def settings(tmp_path):
    return SimpleNamespace(
        MIN_STABILITY_SCORE=0.6,
        MIN_PROFIT_FACTOR=1.0,
        MAX_ALLOWED_DRAWDOWN=0.15,
        PRODUCTION_MODEL_PATH=str(tmp_path / "models" / "best_model.joblib"),
    )


class DummyEstimator:
    def predict(self, X):
        return [1] * len(X)


def test_model_promotion_rejects_overfitting_high(tmp_path):
    model = tmp_path / "candidate.joblib"
    report = tmp_path / "report.json"
    joblib.dump({"model": DummyEstimator(), "features": ["close"]}, model)
    report.write_text(json.dumps({"overfitting_report": {"risk_level": "HIGH"}}), encoding="utf-8")
    registry = ModelRegistry(str(tmp_path / "registry.json"))
    model_id = registry.register_model(str(model), {"profit_factor": 1.2, "max_drawdown": 0.05, "win_rate": 0.6, "breakeven_win_rate": 0.5, "probability_of_profit": 0.7}, "CANDIDATE", validation_report_path=str(report))

    result = ModelPromotionManager(registry, settings(tmp_path)).promote(model_id)

    assert result["promoted"] is False
    assert registry.get_model(model_id)["status"] == "REJECTED"


def test_model_promotion_promotes_valid_model(tmp_path):
    model = tmp_path / "candidate.joblib"
    joblib.dump({"model": DummyEstimator(), "features": ["close"]}, model)
    registry = ModelRegistry(str(tmp_path / "registry.json"))
    model_id = registry.register_model(str(model), {"profit_factor": 1.2, "max_drawdown": 0.05, "win_rate": 0.6, "breakeven_win_rate": 0.5, "probability_of_profit": 0.7, "stability_score": 0.8}, "CANDIDATE")

    result = ModelPromotionManager(registry, settings(tmp_path)).promote(model_id)

    assert result["promoted"] is True
    assert registry.get_current_model()["model_id"] == model_id
