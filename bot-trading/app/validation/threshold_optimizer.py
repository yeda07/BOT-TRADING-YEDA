import json
from pathlib import Path

import pandas as pd

from app.validation._metrics import probability_up, trading_metrics_from_probabilities


class ThresholdOptimizer:
    def __init__(self, min_threshold: float = 0.52, max_threshold: float = 0.75, step: float = 0.01):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.step = step

    def optimize(self, model, X, y, df, settings) -> dict:
        probabilities = probability_up(model, X)
        rows = []
        threshold = self.min_threshold
        while threshold <= self.max_threshold + 1e-9:
            metrics = trading_metrics_from_probabilities(probabilities, y, settings.PAYOUT, round(threshold, 2))
            rows.append({"threshold": round(threshold, 2), **metrics})
            threshold += self.step
        results = pd.DataFrame(rows)
        eligible = results[results["total_trades"] >= settings.MIN_TRADES_FOR_THRESHOLD]
        if eligible.empty:
            eligible = results
        best = eligible.sort_values(["profit_factor", "edge", "net_profit"], ascending=False).iloc[0].to_dict()
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        results.to_csv("data/logs/threshold_optimization.csv", index=False)
        Path("data/logs/best_threshold.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        return {"best_threshold": float(best["threshold"]), "best": best, "results": rows}
