from pathlib import Path

import joblib
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.market.features import FEATURE_COLUMNS, create_features
from app.ml.evaluate import evaluate_classifier, save_model_metrics
from app.ml.promotion_gate import evaluate_demo_promotion
from app.utils.metrics import breakeven_win_rate, max_drawdown, profit_factor


def train_model(
    candles: pd.DataFrame,
    output_path: str | Path = "models/best_model.joblib",
    expiration_candles: int = 1,
    payout: float = 0.87,
    min_confidence: float = 0.58,
    random_state: int = 42,
    metrics_output_path: str | Path = "data/logs/model_metrics.csv",
) -> dict:
    X, y, features = create_features(candles, expiration_candles=expiration_candles)
    if len(X) < 100:
        raise ValueError("At least 100 feature rows are recommended to train the ML model.")

    X_train, X_validation, X_test, y_train, y_validation, y_test = temporal_train_validation_test_split(X, y)
    models = candidate_models(random_state)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        validation_metrics = evaluate_classifier(model, X_validation, y_validation)
        validation_metrics.update(simulate_predictions(model, X_validation, y_validation, payout=payout))
        validation_metrics["total_samples"] = len(X_validation)

        test_metrics = evaluate_classifier(model, X_test, y_test)
        test_metrics.update(simulate_predictions(model, X_test, y_test, payout=payout))
        test_metrics["total_samples"] = len(X_test)
        test_metrics["split"] = "test"
        test_metrics["is_test_split"] = True

        metrics = validation_metrics.copy()
        metrics["split"] = "validation"
        metrics["is_test_split"] = False
        metrics["validation"] = validation_metrics
        metrics["test"] = test_metrics
        results[name] = metrics

    save_model_metrics(results, output_path=metrics_output_path)

    breakeven = breakeven_win_rate(payout)
    best_name = max(results, key=lambda name: _selection_score(results[name], breakeven))
    demo_gate = evaluate_demo_promotion(results[best_name]["test"], breakeven)
    results[best_name]["demo_gate"] = demo_gate
    results[best_name]["eligible_for_demo"] = demo_gate["eligible"]
    best_model = candidate_models(random_state)[best_name]
    best_model.fit(pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation]))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "features": features,
            "best_model": best_name,
            "metrics": results[best_name],
            "all_metrics": results,
            "expiration_candles": expiration_candles,
            "min_confidence": min_confidence,
            "payout": payout,
            "breakeven_win_rate": breakeven,
            "demo_gate": demo_gate,
            "eligible_for_demo": demo_gate["eligible"],
        },
        output_path,
    )
    return {
        "best_model": best_name,
        "models": results,
        "features": features,
        "payout": payout,
        "breakeven_win_rate": breakeven,
        "demo_gate": demo_gate,
        "eligible_for_demo": demo_gate["eligible"],
        "metrics_csv": str(metrics_output_path),
    }


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


def temporal_train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    validation_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    if not 0 < train_size < 1 or not 0 < validation_size < 1:
        raise ValueError("train_size and validation_size must be between 0 and 1.")
    if train_size + validation_size >= 1:
        raise ValueError("train_size + validation_size must leave room for test data.")

    train_end = int(len(X) * train_size)
    validation_end = int(len(X) * (train_size + validation_size))
    if train_end <= 0 or validation_end <= train_end or validation_end >= len(X):
        raise ValueError("Temporal split produced an empty train, validation or test set.")

    return (
        X.iloc[:train_end],
        X.iloc[train_end:validation_end],
        X.iloc[validation_end:],
        y.iloc[:train_end],
        y.iloc[train_end:validation_end],
        y.iloc[validation_end:],
    )


def candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoostingClassifier": Pipeline(
            steps=[
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "ExtraTreesClassifier": Pipeline(
            steps=[
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def simulate_predictions(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, payout: float = 0.87) -> dict[str, float]:
    predictions = model.predict(X_test)
    wins = predictions == y_test.to_numpy()
    pnl = np.where(wins, payout, -1.0)
    pnl_series = pd.Series(pnl)
    equity = 100.0 + pnl_series.cumsum()
    return {
        "win_rate_simulado": float(wins.mean()),
        "profit_simulado": float(pnl.sum()),
        "profit_factor_simulado": float(profit_factor(pnl_series)),
        "total_trades_simulado": int(len(pnl_series)),
        "max_drawdown_simulado": float(max_drawdown(equity)),
        "beats_breakeven": bool(float(wins.mean()) > breakeven_win_rate(payout)),
    }


def _selection_score(metrics: dict, breakeven: float) -> tuple[float, float, int, int, float]:
    roc_auc = metrics.get("roc_auc", 0.0)
    if roc_auc is None or (isinstance(roc_auc, float) and math.isnan(roc_auc)):
        roc_auc = 0.0
    win_rate = metrics.get("win_rate_simulado", 0.0)
    simulated_profit_factor = metrics.get("profit_factor_simulado", 0.0)
    return (
        float(roc_auc),
        float(metrics.get("f1", 0.0)),
        int(win_rate > breakeven),
        int(simulated_profit_factor > 1),
        float(metrics.get("profit_simulado", 0.0)),
    )
