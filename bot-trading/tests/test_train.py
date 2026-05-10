import joblib
import pandas as pd

from app.market.features import FEATURE_COLUMNS
from app.ml.train import temporal_train_test_split, train_model


def sample_candles(rows: int = 260) -> pd.DataFrame:
    prices = []
    price = 100.0
    for index in range(rows):
        price += 0.4 if index % 8 < 4 else -0.3
        prices.append(price)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": [p + (0.1 if index % 2 == 0 else -0.1) for index, p in enumerate(prices)],
            "volume": [100 + index for index in range(rows)],
        }
    )


def test_temporal_train_test_split_keeps_old_data_in_train():
    X = pd.DataFrame({"value": range(10)})
    y = pd.Series([0, 1] * 5)
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.3)

    assert X_train["value"].tolist() == list(range(7))
    assert X_test["value"].tolist() == [7, 8, 9]
    assert y_train.index.max() < y_test.index.min()


def test_train_model_compares_models_and_saves_best_artifact(tmp_path):
    output_path = tmp_path / "best_model.joblib"
    result = train_model(sample_candles(), output_path=output_path, expiration_candles=1)

    assert output_path.exists()
    assert set(result["models"]) == {
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
    }
    for metrics in result["models"].values():
        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "confusion_matrix",
            "win_rate_simulado",
            "profit_simulado",
        ]:
            assert key in metrics

    artifact = joblib.load(output_path)
    assert artifact["features"] == FEATURE_COLUMNS
    assert artifact["best_model"] == result["best_model"]
