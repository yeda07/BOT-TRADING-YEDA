import pandas as pd

from app.execution.ml_backtester import MLBacktester
from app.market.features import FEATURE_COLUMNS, build_features


class ProbModel:
    classes_ = [0, 1]

    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]


def sample_candles(rows: int = 240) -> pd.DataFrame:
    prices = [100 + index * 0.1 for index in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [price + 0.2 for price in prices],
            "low": [price - 0.2 for price in prices],
            "close": [price + 0.05 for price in prices],
            "volume": [100] * rows,
        }
    )


def test_ml_backtester_calculates_metrics_and_saves_csv(tmp_path):
    featured = build_features(sample_candles())
    tester = MLBacktester(
        model=ProbModel(),
        features=FEATURE_COLUMNS,
        initial_balance=1000,
        stake=10,
        payout=0.8,
        min_confidence=0.58,
        output_dir=tmp_path,
    )

    metrics, trades = tester.run(featured)

    assert metrics.total_trades == len(trades)
    assert metrics.final_balance > metrics.initial_balance
    assert metrics.win_rate == 1.0
    assert metrics.profit_factor > 1
    assert (tmp_path / "ml_backtest_results.csv").exists()
    assert (tmp_path / "ml_equity_curve.csv").exists()
