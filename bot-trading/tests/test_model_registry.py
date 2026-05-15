from app.mlops.model_registry import ModelRegistry


def test_model_registry_registers_and_sets_production(tmp_path):
    registry = ModelRegistry(str(tmp_path / "registry.json"))
    model_id = registry.register_model("models/x.joblib", {"score": 1.0}, "CANDIDATE")

    registry.set_status(model_id, "PRODUCTION")

    assert registry.get_current_model()["model_id"] == model_id
    assert registry.get_model(model_id)["status"] == "PRODUCTION"
