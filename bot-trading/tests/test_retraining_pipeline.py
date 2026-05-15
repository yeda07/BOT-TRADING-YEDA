from types import SimpleNamespace

from app.mlops.model_registry import ModelRegistry
from app.mlops.retraining_pipeline import RetrainingPipeline


def test_retraining_pipeline_does_not_run_without_enough_candles(tmp_path):
    settings = SimpleNamespace(
        COLLECTED_CANDLES_PATH=str(tmp_path / "missing.csv"),
        RETRAIN_MIN_NEW_CANDLES=1000,
    )
    result = RetrainingPipeline(settings, ModelRegistry(str(tmp_path / "registry.json"))).run()

    assert result["recommendation"] == "NEEDS_MORE_DATA"
