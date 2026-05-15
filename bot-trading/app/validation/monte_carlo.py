import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.metrics import max_drawdown


class MonteCarloSimulator:
    def __init__(self, trades_df, simulations: int = 1000):
        self.trades_df = trades_df.copy()
        self.simulations = simulations

    def run(self) -> dict:
        pnl = self._pnl()
        rows = []
        if pnl.empty:
            summary = self._empty()
        else:
            rng = np.random.default_rng(42)
            for index in range(self.simulations):
                sample = pd.Series(rng.permutation(pnl.to_numpy()))
                equity = 100.0 + sample.cumsum()
                rows.append({"simulation": index, "profit": float(sample.sum()), "max_drawdown": float(max_drawdown(equity))})
            results = pd.DataFrame(rows)
            summary = {
                "probability_of_ruin": float((results["profit"] <= -100).mean()),
                "expected_drawdown": float(results["max_drawdown"].mean()),
                "worst_drawdown": float(results["max_drawdown"].max()),
                "profit_p5": float(results["profit"].quantile(0.05)),
                "profit_p50": float(results["profit"].quantile(0.50)),
                "profit_p95": float(results["profit"].quantile(0.95)),
                "probability_of_profit": float((results["profit"] > 0).mean()),
            }
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv("data/logs/monte_carlo_results.csv", index=False)
        Path("data/logs/monte_carlo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"summary": summary, "results": rows}

    def _pnl(self) -> pd.Series:
        for column in ["pnl", "profit", "net_profit"]:
            if column in self.trades_df.columns:
                return pd.to_numeric(self.trades_df[column], errors="coerce").dropna()
        return pd.Series(dtype=float)

    def _empty(self) -> dict:
        return {
            "probability_of_ruin": 0.0,
            "expected_drawdown": 0.0,
            "worst_drawdown": 0.0,
            "profit_p5": 0.0,
            "profit_p50": 0.0,
            "profit_p95": 0.0,
            "probability_of_profit": 0.0,
        }
