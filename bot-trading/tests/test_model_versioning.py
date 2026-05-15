from pathlib import Path

import joblib

from app.mlops.model_versioning import copy_to_production, rollback_model, save_versioned_model


def test_model_versioning_creates_copy_and_rollback(tmp_path):
    source = tmp_path / "source.joblib"
    production = tmp_path / "best_model.joblib"
    joblib.dump({"v": 1}, source)

    versioned = save_versioned_model(source, str(tmp_path / "versions"))
    copy_to_production(versioned, str(production))

    assert Path(versioned).exists()
    assert production.exists()

    other = tmp_path / "other.joblib"
    joblib.dump({"v": 2}, other)
    rollback_model(str(other), str(production))
    assert joblib.load(production)["v"] == 2
