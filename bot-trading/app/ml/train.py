from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.market.features import FEATURE_COLUMNS, build_supervised_dataset
from app.ml.evaluate import evaluate_classifier


def train_model(
    candles: pd.DataFrame,
    output_path: str | Path = "models/best_model.joblib",
    expiration_candles: int = 1,
    test_size: float = 0.25,
    payout: float = 0.87,
    random_state: int = 42,
) -> dict:
    X, y = build_supervised_dataset(candles, expiration_candles=expiration_candles)
    if len(X) < 100:
        raise ValueError("At least 100 feature rows are recommended to train the ML model.")

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=test_size)
    models = candidate_models(random_state)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test)
        metrics.update(simulate_predictions(model, X_test, y_test, payout=payout))
        results[name] = metrics

    best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["profit_simulado"]))
    best_model = models[best_name]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "features": FEATURE_COLUMNS,
            "best_model": best_name,
            "metrics": results[best_name],
        },
        output_path,
    )
    return {"best_model": best_name, "models": results, "features": FEATURE_COLUMNS}


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    split_index = int(len(X) * (1 - test_size))
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Temporal split produced an empty train or test set.")
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=250, max_depth=6, random_state=random_state),
                ),
            ]
        ),
        "GradientBoostingClassifier": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def simulate_predictions(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, payout: float = 0.87) -> dict[str, float]:
    predictions = model.predict(X_test)
    wins = predictions == y_test.to_numpy()
    pnl = np.where(wins, payout, -1.0)
    return {
        "win_rate_simulado": float(wins.mean()),
        "profit_simulado": float(pnl.sum()),
    }
