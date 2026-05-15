import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.market.features import create_features
from app.validation._metrics import classification_metrics, probability_up, trading_metrics_from_probabilities


class WalkForwardValidator:
    def __init__(self, df, settings, train_window: int, test_window: int, step_size: int):
        self.df = df
        self.settings = settings
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def run(self) -> dict:
        rows = []
        fold = 0
        max_start = len(self.df) - self.train_window - self.test_window + 1
        for start in range(0, max(0, max_start), self.step_size):
            train_df = self.df.iloc[start : start + self.train_window]
            test_df = self.df.iloc[start + self.train_window : start + self.train_window + self.test_window]
            try:
                X_train, y_train, features = create_features(train_df, self.settings.EXPIRATION_CANDLES)
                X_test, y_test, _ = create_features(test_df, self.settings.EXPIRATION_CANDLES)
            except ValueError:
                continue
            if len(X_train) < 20 or len(X_test) < 5 or len(set(y_train)) < 2:
                continue
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X_train[features], y_train)
            probabilities = probability_up(model, X_test[features])
            trade = trading_metrics_from_probabilities(probabilities, y_test, self.settings.PAYOUT, self.settings.MIN_CONFIDENCE)
            classes = classification_metrics(model, X_test[features], y_test)
            rows.append(
                {
                    "fold": fold,
                    "train_start": str(train_df.iloc[0]["timestamp"]),
                    "train_end": str(train_df.iloc[-1]["timestamp"]),
                    "test_start": str(test_df.iloc[0]["timestamp"]),
                    "test_end": str(test_df.iloc[-1]["timestamp"]),
                    **trade,
                    **classes,
                }
            )
            fold += 1

        results = pd.DataFrame(rows)
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        results.to_csv("data/logs/walk_forward_results.csv", index=False)
        summary = self._summary(results)
        Path("data/logs/walk_forward_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"folds": rows, "summary": summary, "features": locals().get("features", [])}

    def _summary(self, results: pd.DataFrame) -> dict:
        if results.empty:
            return {
                "average_win_rate": 0.0,
                "std_win_rate": 0.0,
                "average_profit_factor": 0.0,
                "average_drawdown": 0.0,
                "profitable_folds": 0,
                "total_folds": 0,
                "profitable_folds_ratio": 0.0,
                "stability_score": 0.0,
            }
        profitable = int((results["net_profit"] > 0).sum())
        avg_pf = float(results["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(3.0).mean())
        std_wr = float(results["win_rate"].std(ddof=0))
        stability = max(0.0, min(1.0, float(results["win_rate"].mean()) - std_wr + min(avg_pf, 2.0) / 4))
        return {
            "average_win_rate": float(results["win_rate"].mean()),
            "std_win_rate": std_wr,
            "average_profit_factor": avg_pf,
            "average_drawdown": float(results["max_drawdown"].mean()),
            "profitable_folds": profitable,
            "total_folds": int(len(results)),
            "profitable_folds_ratio": profitable / len(results),
            "stability_score": stability,
        }
