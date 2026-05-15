from app.validation.walk_forward import WalkForwardValidator
from tests.validation_helpers import sample_candles, validation_settings


def test_walk_forward_validator_generates_valid_folds(tmp_path):
    result = WalkForwardValidator(sample_candles(180), validation_settings(), 90, 60, 30).run()

    assert "summary" in result
    assert result["summary"]["total_folds"] >= 1
    assert result["folds"][0]["train_end"] < result["folds"][0]["test_end"]
