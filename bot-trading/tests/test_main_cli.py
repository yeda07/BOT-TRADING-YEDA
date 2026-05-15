import app.main as main_module
import pandas as pd
import pytest

from app.main import (
    require_cli_csv,
    run_collect_data_cli,
    run_data_quality_cli,
    run_demo_cli,
    run_models_cli,
    run_retrain_cli,
    run_live_paper_cli,
    run_paper_cli,
    run_predict_latest_cli,
    run_validate_cli,
)


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


def test_require_cli_csv_reports_expected_format(tmp_path):
    with pytest.raises(FileNotFoundError, match="timestamp,open,high,low,close,volume"):
        require_cli_csv(str(tmp_path / "missing.csv"))


def test_predict_latest_cli_missing_model_shows_controlled_error(tmp_path):
    csv_path = tmp_path / "candles.csv"
    sample_rows = 240
    prices = [100 + index * 0.1 for index in range(sample_rows)]
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=sample_rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [price + 0.2 for price in prices],
            "low": [price - 0.2 for price in prices],
            "close": [price + 0.05 for price in prices],
            "volume": [100] * sample_rows,
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(FileNotFoundError, match="Run `python -m app.main train` first"):
        run_predict_latest_cli(str(csv_path), str(tmp_path / "missing.joblib"))


def test_live_paper_cli_missing_model_shows_controlled_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `python -m app.main train` first"):
        run_live_paper_cli(str(tmp_path / "candles.csv"), str(tmp_path / "missing.joblib"))


def test_live_paper_cli_missing_candles_shows_controlled_error(tmp_path):
    model_path = tmp_path / "best_model.joblib"
    model_path.write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError, match="Expected data/raw/candles.csv"):
        run_live_paper_cli(str(tmp_path / "missing.csv"), str(model_path))


def test_collect_data_cli_generates_collected_candles(tmp_path, monkeypatch):
    csv_path = tmp_path / "candles.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [100] * 5,
        }
    ).to_csv(csv_path, index=False)
    collected_path = tmp_path / "collected.csv"
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "DATA_FEED_SOURCE": "mock_realtime",
                "CANDLES_CSV_PATH": str(csv_path),
                "COLLECTED_CANDLES_PATH": str(collected_path),
                "LIVE_MAX_STEPS": 3,
            }
        ),
    )

    result = run_collect_data_cli(str(csv_path))

    assert result == str(collected_path)
    assert len(pd.read_csv(collected_path)) == 3


def test_data_quality_cli_prints_report(tmp_path, capsys):
    csv_path = tmp_path / "candles.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC"),
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100, 101],
            "volume": [100] * 2,
            "asset": ["EURUSD-OTC"] * 2,
            "timeframe_seconds": [60] * 2,
            "source": ["test"] * 2,
        }
    ).to_csv(csv_path, index=False)

    run_data_quality_cli(str(csv_path))

    assert "DATA QUALITY REPORT" in capsys.readouterr().out


def test_demo_cli_does_not_allow_real_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(update={"BOT_MODE": "real", "ENABLE_REAL_TRADING": True}),
    )

    with pytest.raises(RuntimeError, match="Real trading is disabled"):
        run_demo_cli(str(tmp_path / "missing.joblib"))


def test_validate_cli_runs_with_synthetic_dataset(tmp_path, monkeypatch):
    csv_path = tmp_path / "candles.csv"
    rows = 180
    prices = [100 + (index % 10) * 0.1 for index in range(rows)]
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [p + 0.3 for p in prices],
            "low": [p - 0.3 for p in prices],
            "close": [p + 0.05 for p in prices],
            "volume": [100] * rows,
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "VALIDATION_TRAIN_WINDOW": 90,
                "VALIDATION_TEST_WINDOW": 60,
                "VALIDATION_STEP_SIZE": 30,
                "VALIDATION_N_SPLITS": 2,
                "MIN_TRADES_FOR_THRESHOLD": 5,
                "MONTE_CARLO_SIMULATIONS": 10,
                "MAX_ALLOWED_DRAWDOWN": 0.5,
            }
        ),
    )

    result = run_validate_cli(str(csv_path))

    assert "report" in result


def test_main_models_lists_models(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "settings", main_module.settings.model_copy(update={"MODEL_REGISTRY_PATH": str(tmp_path / "registry.json")}))

    result = run_models_cli()

    assert result == []


def test_main_retrain_does_not_auto_promote(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "MODEL_REGISTRY_PATH": str(tmp_path / "registry.json"),
                "COLLECTED_CANDLES_PATH": str(tmp_path / "missing.csv"),
                "AUTO_PROMOTE_MODELS": False,
                "RETRAIN_MIN_NEW_CANDLES": 100,
            }
        ),
    )

    result = run_retrain_cli()

    assert result["recommendation"] == "NEEDS_MORE_DATA"
