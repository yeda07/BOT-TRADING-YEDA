import json
from pathlib import Path

from app.mlops.model_versioning import copy_to_production, rollback_model


class ModelPromotionManager:
    def __init__(self, model_registry, settings):
        self.model_registry = model_registry
        self.settings = settings

    def evaluate_candidate(self, candidate_model_id: str) -> dict:
        model = self.model_registry.get_model(candidate_model_id)
        metrics = model.get("metrics", {})
        report = self._load_report(model.get("validation_report_path"))
        overfit = report.get("overfitting_report", {})
        leakage = report.get("data_leakage_audit", {})
        stability = report.get("model_stability_report", {})
        wf = report.get("walk_forward_summary", {})
        mc = report.get("monte_carlo_summary", {})
        ok = (
            not leakage.get("has_critical_leakage", False)
            and overfit.get("risk_level", "LOW") != "HIGH"
            and stability.get("stability_score", metrics.get("stability_score", 0.0)) >= self.settings.MIN_STABILITY_SCORE
            and metrics.get("profit_factor", wf.get("average_profit_factor", 0.0)) >= self.settings.MIN_PROFIT_FACTOR
            and metrics.get("max_drawdown", wf.get("average_drawdown", 1.0)) <= self.settings.MAX_ALLOWED_DRAWDOWN
            and metrics.get("win_rate", wf.get("average_win_rate", 0.0)) > metrics.get("breakeven_win_rate", 0.0)
            and mc.get("probability_of_profit", metrics.get("probability_of_profit", 0.0)) >= 0.60
        )
        return {"approved": bool(ok), "reason": "candidate passed promotion checks" if ok else "candidate failed promotion checks"}

    def promote(self, candidate_model_id: str) -> dict:
        evaluation = self.evaluate_candidate(candidate_model_id)
        if not evaluation["approved"]:
            self.model_registry.set_status(candidate_model_id, "REJECTED")
            return {"promoted": False, **evaluation}
        current = self.model_registry.get_current_model()
        if current:
            self.model_registry.set_status(current["model_id"], "ARCHIVED")
        model = self.model_registry.get_model(candidate_model_id)
        copy_to_production(model["path"], self._production_model_path())
        self.model_registry.set_status(candidate_model_id, "PRODUCTION")
        return {"promoted": True, "model_id": candidate_model_id}

    def rollback(self, model_id: str) -> dict:
        model = self.model_registry.get_model(model_id)
        rollback_model(model["path"], self._production_model_path())
        current = self.model_registry.get_current_model()
        if current:
            self.model_registry.set_status(current["model_id"], "ARCHIVED")
        self.model_registry.set_status(model_id, "PRODUCTION")
        return {"rolled_back": True, "model_id": model_id}

    def _load_report(self, path: str | None) -> dict:
        if path and Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
        return {}

    def _production_model_path(self) -> str:
        return str(getattr(self.settings, "PRODUCTION_MODEL_PATH", "models/best_model.joblib"))
