import joblib
import pandas as pd
import pytest

from app.market.features import FEATURE_COLUMNS
from app.ml.predict import MLPredictor, predict_latest
from app.market.features import build_features


class ProbModel:
    classes_ = [0, 1]

    def __init__(self, probability_up: float) -> None:
        self.probability_up = probability_up

    def predict(self, X):
        return [1 if self.probability_up >= 0.5 else 0]

    def predict_proba(self, X):
        return [[1 - self.probability_up, self.probability_up]]


def sample_candles(rows: int = 240) -> pd.DataFrame:
    prices = [100 + i * 0.1 for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": [p + 0.05 for p in prices],
            "volume": [100] * rows,
        }
    )


def save_model(tmp_path, probability_up: float):
    path = tmp_path / "best_model.joblib"
    joblib.dump({"model": ProbModel(probability_up), "features": FEATURE_COLUMNS}, path)
    return path


def test_predict_latest_returns_buy_when_probability_up_is_high(tmp_path):
    result = predict_latest(sample_candles(), save_model(tmp_path, 0.70), min_confidence=0.58)
    assert result == {"signal": "BUY", "confidence": 0.70, "probability_up": 0.70}


def test_predict_latest_returns_sell_when_probability_up_is_low(tmp_path):
    result = predict_latest(sample_candles(), save_model(tmp_path, 0.30), min_confidence=0.58)
    assert result == {"signal": "SELL", "confidence": 0.70, "probability_up": 0.30}


def test_predict_latest_returns_hold_between_thresholds(tmp_path):
    result = predict_latest(sample_candles(), save_model(tmp_path, 0.50), min_confidence=0.58)
    assert result == {"signal": "HOLD", "confidence": 0.50, "probability_up": 0.50}


def test_ml_predictor_predicts_dataframe(tmp_path):
    predictor = MLPredictor(save_model(tmp_path, 0.70))
    predictions = predictor.predict_dataframe(build_features(sample_candles()))

    assert set(predictions["signal"].unique()) == {"BUY"}
    assert "probability_up" in predictions.columns


def test_ml_predictor_missing_model_has_controlled_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `python -m app.main train` first"):
        MLPredictor(tmp_path / "missing.joblib")
