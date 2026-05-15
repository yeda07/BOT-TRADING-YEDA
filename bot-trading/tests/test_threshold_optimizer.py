from sklearn.ensemble import RandomForestClassifier

from app.market.features import create_features
from app.validation.threshold_optimizer import ThresholdOptimizer
from tests.validation_helpers import sample_candles, validation_settings


def test_threshold_optimizer_selects_threshold_with_minimum_trades():
    df = sample_candles(180)
    X, y, features = create_features(df, expiration_candles=1)
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

    result = ThresholdOptimizer(0.52, 0.60, 0.02).optimize(model, X, y, df, validation_settings(MIN_TRADES_FOR_THRESHOLD=5))

    assert result["best"]["total_trades"] >= 5
    assert 0.52 <= result["best_threshold"] <= 0.60
