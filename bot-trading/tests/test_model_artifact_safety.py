from pathlib import Path

import joblib

from app.ml.predict import MLPredictor
from app.mlops.model_promotion import ModelPromotionManager
from app.mlops.model_registry import ModelRegistry
from app.mlops.model_versioning import save_versioned_model


class DummyEstimator:
    def predict(self, X):
        return [1] * len(X)


class Settings:
    MIN_STABILITY_SCORE = 0.6
    MIN_PROFIT_FACTOR = 1.0
    MAX_ALLOWED_DRAWDOWN = 0.15

    def __init__(self, production_model_path: str):
        self.PRODUCTION_MODEL_PATH = production_model_path


def test_model_operations_do_not_overwrite_real_best_model(tmp_path):
    real_model = Path("models/best_model.joblib")
    before = real_model.read_bytes() if real_model.exists() else None

    candidate = tmp_path / "candidate.joblib"
    joblib.dump({"model": DummyEstimator(), "features": ["close"]}, candidate)

    registry = ModelRegistry(str(tmp_path / "model_registry.json"))
    model_id = registry.register_model(
        str(candidate),
        {
            "profit_factor": 1.2,
            "max_drawdown": 0.05,
            "win_rate": 0.6,
            "breakeven_win_rate": 0.5,
            "probability_of_profit": 0.7,
            "stability_score": 0.8,
        },
        "CANDIDATE",
    )
    save_versioned_model(candidate, str(tmp_path / "versions"))
    MLPredictor(candidate, require_demo_eligible=False)
    ModelPromotionManager(registry, Settings(str(tmp_path / "production" / "best_model.joblib"))).promote(model_id)

    if before is not None:
        assert real_model.read_bytes() == before
        artifact = joblib.load(real_model)
        if isinstance(artifact, dict):
            assert not isinstance(artifact.get("model"), str)
