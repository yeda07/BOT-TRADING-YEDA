import json
from pathlib import Path

import pandas as pd


class ModelStabilityAnalyzer:
    def analyze(self, walk_forward_results, threshold_results, stress_results) -> dict:
        wf = pd.DataFrame(walk_forward_results)
        thresholds = pd.DataFrame(threshold_results)
        stress = pd.DataFrame(stress_results)
        if wf.empty:
            score = 0.0
            reason = "No walk-forward folds available."
        else:
            win_stability = max(0.0, 1 - float(wf["win_rate"].std(ddof=0)))
            pf_stability = min(1.0, float(wf["profit_factor"].replace(float("inf"), 3).mean()) / 2)
            dd_stability = max(0.0, 1 - float(wf["max_drawdown"].mean()))
            stress_ratio = float(stress["survived"].mean()) if not stress.empty and "survived" in stress else 0.0
            threshold_depth = 1.0 if thresholds.empty else min(1.0, float(thresholds["total_trades"].max()) / 100)
            score = 0.30 * win_stability + 0.25 * pf_stability + 0.20 * dd_stability + 0.15 * stress_ratio + 0.10 * threshold_depth
            reason = "Stability score computed from walk-forward, threshold and stress results."
        status = "STABLE" if score >= 0.70 else "UNSTABLE" if score >= 0.40 else "REJECTED"
        result = {"stability_score": float(score), "status": status, "reason": reason}
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/logs/model_stability_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
