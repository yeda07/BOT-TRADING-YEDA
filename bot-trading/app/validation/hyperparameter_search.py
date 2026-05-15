from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.validation._metrics import classification_metrics, composite_score, probability_up, trading_metrics_from_probabilities
from app.validation.time_series_split import SafeTimeSeriesSplit


class HyperparameterSearch:
    def __init__(self, settings):
        self.settings = settings
        self.uses_shuffle = False

    def run(self, X, y, features) -> dict:
        splitter = SafeTimeSeriesSplit(
            n_splits=min(self.settings.VALIDATION_N_SPLITS, 3),
            gap=self.settings.VALIDATION_GAP,
            min_train_size=min(max(50, len(X) // 2), max(1, len(X) - 20)),
        )
        candidates = self._models()
        rows = []
        best = None
        best_score = -999.0
        for name, model in candidates:
            fold_scores = []
            for train_idx, test_idx in splitter.split(X):
                model.fit(X.iloc[train_idx][features], y.iloc[train_idx])
                probabilities = probability_up(model, X.iloc[test_idx][features])
                metrics = {
                    **classification_metrics(model, X.iloc[test_idx][features], y.iloc[test_idx]),
                    **trading_metrics_from_probabilities(probabilities, y.iloc[test_idx], self.settings.PAYOUT, self.settings.MIN_CONFIDENCE),
                }
                fold_scores.append(composite_score(metrics))
            score = float(sum(fold_scores) / len(fold_scores))
            rows.append({"model": name, "score": score, "uses_shuffle": False})
            if score > best_score:
                best_score = score
                best = (name, model)

        best_name, best_model = best
        best_model.fit(X[features], y)
        Path("models").mkdir(exist_ok=True)
        joblib.dump({"model": best_model, "features": features, "best_model": best_name, "score": best_score}, "models/optimized_model.joblib")
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv("data/logs/hyperparameter_results.csv", index=False)
        return {"best_model": best_name, "best_score": best_score, "results": rows, "uses_shuffle": False}

    def _models(self):
        return [
            ("LogisticRegression", Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=500, random_state=42))])),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)),
            ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
        ]
