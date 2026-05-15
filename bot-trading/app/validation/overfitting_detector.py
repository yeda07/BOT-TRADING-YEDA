import json
from pathlib import Path


class OverfittingDetector:
    def analyze(self, train_metrics: dict, validation_metrics: dict, test_metrics: dict, walk_forward_summary: dict) -> dict:
        warnings = []
        recommendations = []
        if train_metrics.get("accuracy", 0) - test_metrics.get("accuracy", 0) > 0.10:
            warnings.append("Train accuracy is much higher than test accuracy.")
        if train_metrics.get("roc_auc", 0) - test_metrics.get("roc_auc", 0) > 0.10:
            warnings.append("Train ROC AUC is much higher than test ROC AUC.")
        if walk_forward_summary.get("profitable_folds_ratio", 0) < 0.5:
            warnings.append("Walk-forward profitable fold ratio is weak.")
        if test_metrics.get("total_trades", test_metrics.get("total_trades_simulado", 0)) < 50 and test_metrics.get("win_rate", test_metrics.get("win_rate_simulado", 0)) > 0.7:
            warnings.append("High win rate is based on too few trades.")
        if walk_forward_summary.get("std_win_rate", 0) > 0.12:
            warnings.append("High variance between walk-forward folds.")
        if walk_forward_summary.get("average_drawdown", 0) > 0.15:
            warnings.append("Drawdown is unstable or too high.")

        risk = "LOW"
        if len(warnings) >= 4:
            risk = "HIGH"
        elif warnings:
            risk = "MEDIUM"
        if warnings:
            recommendations.append("Keep the system in paper/demo validation and gather more out-of-sample data.")
        else:
            recommendations.append("Continue paper/demo validation; never infer real-money readiness from this alone.")
        result = {"is_overfitted": risk == "HIGH", "risk_level": risk, "warnings": warnings, "recommendations": recommendations}
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/logs/overfitting_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
