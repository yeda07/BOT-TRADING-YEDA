from pathlib import Path

import joblib
import pandas as pd

from app.market.features import FEATURE_COLUMNS
from app.strategies.rule_based import StrategySignal


class MLStrategy:
    def __init__(self, model_path: str | Path, min_confidence: float = 0.60) -> None:
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
        self.features = artifact.get("features", FEATURE_COLUMNS) if isinstance(artifact, dict) else FEATURE_COLUMNS

    def generate_signal(self, row: pd.Series) -> StrategySignal:
        if row[self.features].isna().any():
            return StrategySignal("HOLD", 0.0, "Insufficient ML features.")

        X = pd.DataFrame([row[self.features].to_dict()])
        prediction = int(self.model.predict(X)[0])
        confidence = self._confidence(X, prediction)
        if confidence < self.min_confidence:
            return StrategySignal("HOLD", confidence, "Low model confidence.")
        return StrategySignal("BUY" if prediction == 1 else "SELL", confidence, "ML prediction.")

    def _confidence(self, X: pd.DataFrame, prediction: int) -> float:
        if not hasattr(self.model, "predict_proba"):
            return 0.5
        classes = list(self.model.classes_)
        probabilities = self.model.predict_proba(X)[0]
        return float(probabilities[classes.index(prediction)])
