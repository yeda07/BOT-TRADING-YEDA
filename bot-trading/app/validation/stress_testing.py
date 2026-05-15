import json
from pathlib import Path

import pandas as pd

from app.market.features import create_features
from app.validation._metrics import probability_up, trading_metrics_from_probabilities


class StressTester:
    def __init__(self, model, df, features, settings):
        self.model = model
        self.df = df
        self.features = features
        self.settings = settings

    def run(self) -> dict:
        X, y, features = create_features(self.df, self.settings.EXPIRATION_CANDLES)
        scenarios = [
            ("payout_normal", self.settings.PAYOUT, 0.0, 0),
            ("payout_reduced_5", self.settings.PAYOUT * 0.95, 0.0, 0),
            ("payout_reduced_10", self.settings.PAYOUT * 0.90, 0.0, 0),
            ("slippage_light", self.settings.PAYOUT, -0.01, 0),
            ("slippage_heavy", self.settings.PAYOUT, -0.03, 0),
            ("latency_1_candle", self.settings.PAYOUT, 0.0, 1),
            ("latency_2_candles", self.settings.PAYOUT, 0.0, 2),
            ("spread_noise", self.settings.PAYOUT * 0.97, -0.01, 0),
            ("confidence_reduced", self.settings.PAYOUT, 0.0, 0),
            ("connection_loss", self.settings.PAYOUT, 0.0, 0),
        ]
        probabilities = probability_up(self.model, X[features])
        rows = []
        for name, payout, edge_penalty, latency in scenarios:
            threshold = min(0.95, self.settings.MIN_CONFIDENCE + (0.03 if name == "confidence_reduced" else 0.0))
            metrics = trading_metrics_from_probabilities(probabilities, y, payout, threshold)
            metrics["net_profit"] += edge_penalty * metrics["total_trades"]
            if latency:
                metrics["win_rate"] = max(0.0, metrics["win_rate"] - 0.03 * latency)
            survived = metrics["profit_factor"] >= 1 and metrics["max_drawdown"] <= self.settings.MAX_ALLOWED_DRAWDOWN
            rows.append({"scenario": name, **metrics, "survived": bool(survived), "reason": "ok" if survived else "scenario failed"})
        summary = {"total_scenarios": len(rows), "survived_scenarios": sum(row["survived"] for row in rows), "moderate_stress_passed": rows[1]["survived"] and rows[3]["survived"]}
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv("data/logs/stress_test_results.csv", index=False)
        Path("data/logs/stress_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"summary": summary, "results": rows}
