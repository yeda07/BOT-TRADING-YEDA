import joblib
import pandas as pd

from app.execution.comparison import compare_rule_vs_ml
from app.market.features import FEATURE_COLUMNS
from app.risk.risk_manager import RiskManager


class DummyModel:
    classes_ = [-1, 0, 1]

    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        return [[0.05, 0.05, 0.90] for _ in range(len(X))]


def test_compare_rule_vs_ml_returns_both_metrics(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": DummyModel(), "features": FEATURE_COLUMNS, "min_confidence": 0.58}, model_path)
    rows = 120
    prices = [100 + i * 0.2 for i in range(rows)]
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.4 for p in prices],
            "low": [p - 0.4 for p in prices],
            "close": [p + 0.1 for p in prices],
            "volume": [100] * rows,
        }
    )

    comparison, trades = compare_rule_vs_ml(
        candles=candles,
        model_path=model_path,
        risk_manager_factory=lambda: RiskManager(min_model_confidence=0.5),
        initial_balance=1000,
        output_path=tmp_path / "strategy_comparison.csv",
    )

    result = comparison.to_dict()
    assert "rule_based" in result
    assert "ml" in result
    assert comparison.winner in {"rule_based", "ml", "tie"}
    assert set(trades["strategy"].unique()) <= {"rule_based", "ml"}
    assert (tmp_path / "strategy_comparison.csv").exists()


def test_compare_rule_vs_ml_generates_expected_columns(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": DummyModel(), "features": FEATURE_COLUMNS, "min_confidence": 0.58}, model_path)
    rows = 240
    prices = [100 + i * 0.1 for i in range(rows)]
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.3 for p in prices],
            "low": [p - 0.3 for p in prices],
            "close": [p + 0.05 for p in prices],
            "volume": [100] * rows,
        }
    )
    output_path = tmp_path / "strategy_comparison.csv"

    compare_rule_vs_ml(
        candles=candles,
        model_path=model_path,
        risk_manager_factory=lambda: RiskManager(min_model_confidence=0.5),
        initial_balance=1000,
        output_path=output_path,
    )

    report = pd.read_csv(output_path)
    assert list(report.columns) == [
        "strategy",
        "total_trades",
        "wins",
        "losses",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "net_profit",
        "final_balance",
    ]
