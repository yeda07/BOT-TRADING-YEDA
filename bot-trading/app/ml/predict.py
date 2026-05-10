from pathlib import Path

import joblib
import pandas as pd

from app.config import get_settings
from app.market.features import FEATURE_COLUMNS, build_features


def predict_latest(
    candles: pd.DataFrame,
    model_path: str | Path = "models/best_model.joblib",
    min_confidence: float | None = None,
) -> dict[str, float | str]:
    settings = get_settings()
    min_confidence = settings.MIN_CONFIDENCE if min_confidence is None else min_confidence
    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    features = artifact.get("features", FEATURE_COLUMNS) if isinstance(artifact, dict) else FEATURE_COLUMNS
    featured = build_features(candles).dropna()
    if featured.empty:
        raise ValueError("No usable feature rows for prediction.")
    latest = featured.iloc[-1]
    X = pd.DataFrame([latest[features].to_dict()])

    probability_up = _probability_up(model, X)
    if probability_up >= min_confidence:
        signal = "BUY"
        confidence = probability_up
    elif probability_up <= 1 - min_confidence:
        signal = "SELL"
        confidence = 1 - probability_up
    else:
        signal = "HOLD"
        confidence = max(probability_up, 1 - probability_up)

    return {"signal": signal, "confidence": float(confidence), "probability_up": float(probability_up)}


def _probability_up(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(X)[0]
        if 1 in classes:
            return float(probabilities[classes.index(1)])
    return float(int(model.predict(X)[0]))
