from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = _positive_probabilities(model, X_test)
    return classification_metrics(y_test, predictions, probabilities)


def classification_metrics(
    y_true: pd.Series,
    predictions,
    probabilities=None,
) -> dict:
    roc_auc = float("nan")
    if probabilities is not None and pd.Series(y_true).nunique() > 1:
        roc_auc = float(roc_auc_score(y_true, probabilities))

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, predictions, labels=[0, 1], zero_division=0, output_dict=True),
    }


def save_model_metrics(metrics_by_model: dict[str, dict], output_path: str | Path = "data/logs/model_metrics.csv") -> pd.DataFrame:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "roc_auc": metrics.get("roc_auc", float("nan")),
                "total_samples": metrics.get("total_samples", 0),
            }
        )
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _positive_probabilities(model, X: pd.DataFrame):
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    probabilities = model.predict_proba(X)
    if 1 not in classes:
        return None
    return probabilities[:, classes.index(1)]
