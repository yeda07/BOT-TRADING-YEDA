from app.market.candles import load_candles_csv, validate_candles
from app.market.sample_data import ensure_demo_candles_csv, generate_sample_candles


def test_generate_sample_candles_is_valid():
    candles = generate_sample_candles(rows=240)

    validate_candles(candles, min_rows=200)
    assert list(candles.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_ensure_demo_candles_csv_creates_valid_file(tmp_path):
    csv_path = tmp_path / "candles.csv"

    ensured_path = ensure_demo_candles_csv(csv_path, min_rows=200)
    candles = load_candles_csv(ensured_path)

    assert ensured_path == csv_path
    validate_candles(candles, min_rows=200)
