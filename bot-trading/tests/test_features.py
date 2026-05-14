import pandas as pd
import pytest

from app.market.features import FEATURE_COLUMNS, build_features, build_supervised_dataset, create_features


def sample_candles(rows: int = 240) -> pd.DataFrame:
    prices = [100 + i * 0.1 for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": [p + 0.05 for p in prices],
            "volume": [100] * rows,
        }
    )


def test_build_features_creates_ml_columns_without_nan():
    features = build_features(sample_candles())
    assert not features.empty
    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert not features[FEATURE_COLUMNS].isna().any().any()


def test_create_features_returns_X_y_and_feature_names():
    X, y, features = create_features(sample_candles(), expiration_candles=2)

    assert list(X.columns) == features == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert not y.isna().any()
    assert {"ema_ratio", "bb_width", "upper_wick", "lower_wick", "return_10", "momentum_5"} <= set(features)


def test_build_supervised_dataset_uses_binary_target_and_expiration():
    X, y = build_supervised_dataset(sample_candles(), expiration_candles=3)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)
    assert len(X) < len(build_features(sample_candles()))
    assert y.iloc[0] == 1


def test_build_supervised_dataset_rejects_invalid_expiration():
    with pytest.raises(ValueError, match="expiration_candles"):
        build_supervised_dataset(sample_candles(), expiration_candles=0)
