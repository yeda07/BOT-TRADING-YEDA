from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.market.data_quality import validate_candles_df
from app.ml.train import train_model
from app.mlops.model_versioning import save_versioned_model


class RetrainingPipeline:
    def __init__(self, settings, model_registry):
        self.settings = settings
        self.model_registry = model_registry

    def run(self) -> dict:
        data_path = Path(self.settings.COLLECTED_CANDLES_PATH)
        if not data_path.exists():
            result = self._result("", "", 0.0, 0.0, "NEEDS_MORE_DATA", "Collected candles file not found.")
            self._log(result)
            return result
        candles = pd.read_csv(data_path)
        if "timestamp" in candles.columns:
            candles["timestamp"] = pd.to_datetime(candles["timestamp"], errors="coerce")
        valid, errors = validate_candles_df(candles)
        if not valid:
            result = self._result("", "", 0.0, 0.0, "REJECTED", "; ".join(errors))
            self._log(result)
            return result
        if len(candles) < self.settings.RETRAIN_MIN_NEW_CANDLES:
            result = self._result("", "", 0.0, 0.0, "NEEDS_MORE_DATA", "Not enough new candles.")
            self._log(result)
            return result

        candidate_path = "models/candidate_model.joblib"
        metrics = train_model(candles, output_path=candidate_path, expiration_candles=self.settings.EXPIRATION_CANDLES, payout=self.settings.PAYOUT)
        version_path = save_versioned_model(candidate_path, self.settings.MODEL_VERSIONS_DIR)
        candidate_score = _score(metrics)
        current = self.model_registry.get_current_model()
        current_score = float(current.get("metrics", {}).get("score", 0.0)) if current else 0.0
        model_id = self.model_registry.register_model(version_path, {"score": candidate_score, **metrics}, "CANDIDATE")
        recommendation = "CANDIDATE_READY" if candidate_score > current_score else "KEEP_CURRENT"
        result = self._result(model_id, current.get("model_id", ""), candidate_score, current_score, recommendation, "Manual promotion required.")
        self._log(result)
        return result

    def _result(self, candidate_model_id: str, current_model_id: str, candidate_score: float, current_score: float, recommendation: str, reason: str) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidate_model_id": candidate_model_id,
            "current_model_id": current_model_id,
            "candidate_score": candidate_score,
            "current_score": current_score,
            "recommendation": recommendation,
            "reason": reason,
        }

    def _log(self, result: dict) -> None:
        path = Path("data/logs/retraining_runs.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([result])
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _score(metrics: dict) -> float:
    gate = metrics.get("demo_gate", {})
    observed = gate.get("observed", {})
    return float(observed.get("edge_over_breakeven", 0.0)) + min(float(observed.get("profit_factor", 0.0)), 3.0) / 3.0
