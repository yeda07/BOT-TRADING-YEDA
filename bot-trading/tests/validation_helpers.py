from types import SimpleNamespace

import pandas as pd


def sample_candles(rows: int = 180) -> pd.DataFrame:
    prices = []
    price = 100.0
    for index in range(rows):
        price += 0.2 if index % 6 < 3 else -0.1
        prices.append(price)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.3 for p in prices],
            "low": [p - 0.3 for p in prices],
            "close": [p + (0.05 if i % 2 == 0 else -0.05) for i, p in enumerate(prices)],
            "volume": [100 + i for i in range(rows)],
        }
    )


def validation_settings(**updates):
    base = {
        "EXPIRATION_CANDLES": 1,
        "PAYOUT": 0.87,
        "MIN_CONFIDENCE": 0.58,
        "VALIDATION_TRAIN_WINDOW": 90,
        "VALIDATION_TEST_WINDOW": 60,
        "VALIDATION_STEP_SIZE": 30,
        "VALIDATION_N_SPLITS": 2,
        "VALIDATION_GAP": 1,
        "MIN_TRADES_FOR_THRESHOLD": 5,
        "MAX_ALLOWED_DRAWDOWN": 0.50,
        "MONTE_CARLO_SIMULATIONS": 20,
        "MIN_PROFITABLE_FOLDS_RATIO": 0.50,
        "MIN_STABILITY_SCORE": 0.30,
    }
    base.update(updates)
    return SimpleNamespace(**base)
