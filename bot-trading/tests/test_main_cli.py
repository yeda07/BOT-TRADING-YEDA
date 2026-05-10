import pandas as pd

from app.main import run_paper_cli


def test_run_paper_cli_uses_csv_and_paper_broker(tmp_path):
    rows = 240
    prices = [100 + index * 0.1 for index in range(rows)]
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [price + 0.2 for price in prices],
            "low": [price - 0.2 for price in prices],
            "close": [price + 0.05 for price in prices],
            "volume": [100] * rows,
        }
    )
    csv_path = tmp_path / "candles.csv"
    candles.to_csv(csv_path, index=False)

    result = run_paper_cli(str(csv_path))

    assert result["mode"] == "paper"
    assert result["status"] in {"accepted", "blocked", "rejected"}
    assert result["balance"] > 0
