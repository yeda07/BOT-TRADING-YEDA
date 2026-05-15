import json
from pathlib import Path

from app.validation.validation_report import ValidationReportBuilder


def test_validation_report_builder_generates_json_and_markdown(tmp_path, monkeypatch):
    logs = Path("data/logs")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "walk_forward_summary.json").write_text(json.dumps({"total_folds": 1, "profitable_folds_ratio": 1, "average_profit_factor": 1.2, "average_win_rate": 0.6, "average_drawdown": 0.05}), encoding="utf-8")
    (logs / "monte_carlo_summary.json").write_text(json.dumps({"probability_of_profit": 0.7}), encoding="utf-8")
    (logs / "stress_test_summary.json").write_text(json.dumps({"moderate_stress_passed": True}), encoding="utf-8")
    (logs / "overfitting_report.json").write_text(json.dumps({"risk_level": "LOW"}), encoding="utf-8")
    (logs / "model_stability_report.json").write_text(json.dumps({"stability_score": 0.8, "status": "STABLE"}), encoding="utf-8")
    (logs / "data_leakage_audit.json").write_text(json.dumps({"has_critical_leakage": False}), encoding="utf-8")

    result = ValidationReportBuilder().build()

    assert result["final_recommendation"] in {"DEMO_ALLOWED", "PAPER_ONLY", "NEEDS_MORE_DATA", "REJECTED"}
    assert (logs / "final_validation_report.json").exists()
    assert (logs / "final_validation_report.md").exists()
