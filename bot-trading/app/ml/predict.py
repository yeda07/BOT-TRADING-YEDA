from pathlib import Path

import joblib
import pandas as pd

from app.config import get_settings
from app.market.features import FEATURE_COLUMNS, build_features
from app.ml.promotion_gate import assert_artifact_eligible_for_demo


class MLPredictor:
    def __init__(self, model_path: str | Path = "models/best_model.joblib") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}. Run `python -m app.main train` first.")
        artifact = joblib.load(self.model_path)
        settings = get_settings()
        if settings.BOT_MODE == "demo":
            assert_artifact_eligible_for_demo(artifact)
        self.model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
        self.features = artifact.get("features", FEATURE_COLUMNS) if isinstance(artifact, dict) else FEATURE_COLUMNS
        self.min_confidence = artifact.get("min_confidence", settings.MIN_CONFIDENCE) if isinstance(artifact, dict) else settings.MIN_CONFIDENCE

    def predict_row(self, row: pd.Series) -> dict[str, float | str]:
        missing = [feature for feature in self.features if feature not in row.index]
        if missing:
            raise ValueError(f"Prediction row is missing required features: {missing}")
        if row[self.features].isna().any():
            raise ValueError("Prediction row contains NaN feature values.")

        X = pd.DataFrame([row[self.features].to_dict()])
        probability_up = _probability_up(self.model, X)
        if probability_up >= self.min_confidence:
            signal = "BUY"
            confidence = probability_up
        elif probability_up <= 1 - self.min_confidence:
            signal = "SELL"
            confidence = 1 - probability_up
        else:
            signal = "HOLD"
            confidence = max(probability_up, 1 - probability_up)
        return {"signal": signal, "confidence": float(confidence), "probability_up": float(probability_up)}

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        featured = df.copy()
        if not set(self.features).issubset(featured.columns):
            featured = build_features(featured)
        if featured.empty:
            raise ValueError("No usable feature rows for prediction.")
        predictions = [self.predict_row(row) for _, row in featured.iterrows()]
        return pd.concat([featured.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)


def predict_latest(
    candles: pd.DataFrame,
    model_path: str | Path = "models/best_model.joblib",
    min_confidence: float | None = None,
) -> dict[str, float | str]:
    predictor = MLPredictor(model_path)
    if min_confidence is not None:
        predictor.min_confidence = min_confidence
    featured = build_features(candles).dropna()
    if featured.empty:
        raise ValueError("No usable feature rows for prediction.")
    return predictor.predict_row(featured.iloc[-1])


def _probability_up(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(X)[0]
        if 1 in classes:
            return float(probabilities[classes.index(1)])
    return float(int(model.predict(X)[0]))
