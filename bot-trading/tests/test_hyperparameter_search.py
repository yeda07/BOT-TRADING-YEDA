from app.market.features import create_features
from app.validation.hyperparameter_search import HyperparameterSearch
from tests.validation_helpers import sample_candles, validation_settings


def test_hyperparameter_search_does_not_use_shuffle():
    X, y, features = create_features(sample_candles(180), expiration_candles=1)
    result = HyperparameterSearch(validation_settings()).run(X, y, features)

    assert result["uses_shuffle"] is False
    assert result["best_model"]
