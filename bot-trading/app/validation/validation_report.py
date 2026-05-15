import json
from pathlib import Path

import pandas as pd


class ValidationReportBuilder:
    def build(self) -> dict:
        logs = Path("data/logs")
        report = {
            "model_metrics": self._read_csv(logs / "model_metrics.csv"),
            "walk_forward_summary": self._read_json(logs / "walk_forward_summary.json"),
            "threshold_optimization": self._read_csv(logs / "threshold_optimization.csv"),
            "monte_carlo_summary": self._read_json(logs / "monte_carlo_summary.json"),
            "stress_test_summary": self._read_json(logs / "stress_test_summary.json"),
            "overfitting_report": self._read_json(logs / "overfitting_report.json"),
            "model_stability_report": self._read_json(logs / "model_stability_report.json"),
            "data_leakage_audit": self._read_json(logs / "data_leakage_audit.json"),
        }
        report["final_recommendation"] = self._recommend(report)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "final_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (logs / "final_validation_report.md").write_text(self._markdown(report), encoding="utf-8")
        return report

    def _recommend(self, report: dict) -> str:
        wf = report["walk_forward_summary"]
        mc = report["monte_carlo_summary"]
        stress = report["stress_test_summary"]
        overfit = report["overfitting_report"]
        stability = report["model_stability_report"]
        leakage = report["data_leakage_audit"]
        if leakage.get("has_critical_leakage"):
            return "REJECTED"
        if not wf or wf.get("total_folds", 0) == 0:
            return "NEEDS_MORE_DATA"
        if overfit.get("risk_level") == "HIGH" or stability.get("status") == "REJECTED":
            return "REJECTED"
        if (
            wf.get("profitable_folds_ratio", 0) >= 0.60
            and wf.get("average_profit_factor", 0) > 1
            and wf.get("average_win_rate", 0) > 0
            and wf.get("average_drawdown", 1) <= 0.15
            and stability.get("stability_score", 0) >= 0.60
            and mc.get("probability_of_profit", 0) >= 0.60
            and stress.get("moderate_stress_passed", False)
            and overfit.get("risk_level") != "HIGH"
        ):
            return "DEMO_ALLOWED"
        return "PAPER_ONLY"

    def _markdown(self, report: dict) -> str:
        return (
            "# Final Validation Report\n\n"
            "## Dataset summary\nGenerated from configured validation artifacts.\n\n"
            "## Model selected\nSee `model_metrics.csv` and optimized model artifacts.\n\n"
            "## Classification metrics\nIncluded in model metrics and walk-forward folds.\n\n"
            "## Trading metrics\nIncluded in walk-forward, threshold and stress outputs.\n\n"
            f"## Walk-forward results\n{report['walk_forward_summary']}\n\n"
            f"## Threshold chosen\nSee threshold optimization results.\n\n"
            f"## Monte Carlo risk\n{report['monte_carlo_summary']}\n\n"
            f"## Stress testing results\n{report['stress_test_summary']}\n\n"
            f"## Overfitting analysis\n{report['overfitting_report']}\n\n"
            f"## Data leakage audit\n{report['data_leakage_audit']}\n\n"
            f"## Final recommendation\n{report['final_recommendation']}\n"
        )

    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _read_csv(self, path: Path) -> list[dict]:
        return pd.read_csv(path).to_dict("records") if path.exists() else []
