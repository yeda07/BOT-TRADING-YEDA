from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.market.features import build_supervised_dataset
from app.ml.evaluate import evaluate_classifier


def train_model(
    candles: pd.DataFrame,
    output_path: str | Path = "models/model.joblib",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, float]:
    X, y = build_supervised_dataset(candles)
    if len(X) < 100:
        raise ValueError("At least 100 feature rows are recommended to train the ML model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=250, max_depth=6, random_state=random_state)),
        ]
    )
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return metrics

