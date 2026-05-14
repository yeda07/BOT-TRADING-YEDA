from pathlib import Path

import numpy as np
import pandas as pd

from app.market.candles import load_candles_csv, validate_candles


def generate_sample_candles(rows: int = 360, seed: int = 42) -> pd.DataFrame:
    """Build deterministic demo candles for first-run backtests and training."""
    if rows < 200:
        raise ValueError("rows must be at least 200 for the demo dataset.")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    closes: list[float] = []
    previous_close = 100.0

    for index in range(rows):
        regime = 0.015 if (index // 70) % 2 == 0 else -0.015
        cycle = 0.03 * np.sin(index / 5)
        noise = rng.normal(0, 0.06)
        previous_close = max(1.0, previous_close + regime + cycle + noise)
        closes.append(previous_close)

    close = np.array(closes)
    open_ = np.concatenate([[100.0], close[:-1]]) + rng.normal(0, 0.02, rows)
    spread = rng.uniform(0.04, 0.18, rows)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(900, 1800, rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.round(open_, 5),
            "high": np.round(high, 5),
            "low": np.round(low, 5),
            "close": np.round(close, 5),
            "volume": volume,
        }
    )


def ensure_demo_candles_csv(path: str | Path, min_rows: int = 200) -> Path:
    """Create a functional demo CSV when the default dataset is missing or empty."""
    csv_path = Path(path)
    try:
        candles = load_candles_csv(csv_path)
        validate_candles(candles, min_rows=min_rows)
        return csv_path
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generate_sample_candles(rows=max(360, min_rows + 160)).to_csv(csv_path, index=False)
        return csv_path
