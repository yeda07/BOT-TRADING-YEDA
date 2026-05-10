from pathlib import Path

import joblib
import pandas as pd

from app.market.features import FEATURE_COLUMNS, build_features


def predict_latest(candles: pd.DataFrame, model_path: str | Path = "models/model.joblib") -> dict[str, float | str]:
    model = joblib.load(model_path)
    featured = build_features(candles).dropna()
    if featured.empty:
        raise ValueError("No usable feature rows for prediction.")
    latest = featured.iloc[-1]
    X = pd.DataFrame([latest[FEATURE_COLUMNS].to_dict()])
    prediction = int(model.predict(X)[0])
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        confidence = float(model.predict_proba(X)[0][classes.index(prediction)])
    signal = "BUY" if prediction == 1 else "SELL"
    return {"signal": signal, "confidence": confidence}
