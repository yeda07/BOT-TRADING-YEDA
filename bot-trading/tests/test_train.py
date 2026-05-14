import joblib
import pandas as pd

from app.market.features import FEATURE_COLUMNS
from app.ml.train import temporal_train_test_split, temporal_train_validation_test_split, train_model


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


def test_temporal_train_validation_test_split_keeps_order():
    X = pd.DataFrame({"value": range(100)})
    y = pd.Series([0, 1] * 50)

    X_train, X_validation, X_test, y_train, y_validation, y_test = temporal_train_validation_test_split(X, y)

    assert X_train["value"].tolist() == list(range(70))
    assert X_validation["value"].tolist() == list(range(70, 85))
    assert X_test["value"].tolist() == list(range(85, 100))
    assert y_train.index.max() < y_validation.index.min() < y_test.index.min()


def test_train_model_compares_models_and_saves_best_artifact(tmp_path):
    output_path = tmp_path / "best_model.joblib"
    result = train_model(sample_candles(), output_path=output_path, expiration_candles=1)

    assert output_path.exists()
    assert set(result["models"]) == {
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "ExtraTreesClassifier",
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
            "profit_factor_simulado",
            "total_trades_simulado",
            "max_drawdown_simulado",
        ]:
            assert key in metrics

    artifact = joblib.load(output_path)
    assert artifact["features"] == FEATURE_COLUMNS
    assert artifact["best_model"] == result["best_model"]
    assert artifact["expiration_candles"] == 1
    assert "breakeven_win_rate" in artifact
    assert "demo_gate" in artifact
    assert artifact["eligible_for_demo"] is result["eligible_for_demo"]
    assert result["demo_gate"]["observed"]["split"] == "test"
