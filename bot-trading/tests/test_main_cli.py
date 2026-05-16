import app.main as main_module
import joblib
import pandas as pd
import pytest

from app.main import (
    require_cli_csv,
    run_collect_data_cli,
    run_data_quality_cli,
    run_demo_cli,
    run_models_cli,
    run_paper_session_cli,
    run_register_current_model_cli,
    run_feed_status_cli,
    run_append_candles_cli,
    run_reset_feed_cursor_cli,
    run_runtime_status_cli,
    run_sessions_cli,
    run_summary_current_cli,
    run_summary_session_cli,
    run_retrain_cli,
    run_live_paper_cli,
    run_paper_cli,
    run_predict_latest_cli,
    run_validate_cli,
)


class DummyEstimator:
    def predict(self, X):
        return [1] * len(X)


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
                "FEED_CURSOR_PATH": str(tmp_path / "feed_cursor.json"),
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


def test_run_paper_session_blocks_when_kill_switch_active(tmp_path, monkeypatch, capsys):
    kill_path = tmp_path / "kill.json"
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "BOT_MODE": "paper",
                "KILL_SWITCH_PATH": str(kill_path),
                "RUNTIME_STATE_PATH": str(tmp_path / "runtime.json"),
            }
        ),
    )
    main_module.KillSwitch(str(kill_path)).activate("manual")

    run_paper_session_cli()

    assert "Kill switch is active" in capsys.readouterr().out


def test_run_paper_session_warns_when_steps_below_feature_window(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "candles.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="min", tz="UTC"),
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
            "volume": [100] * 20,
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "BOT_MODE": "paper",
                "CANDLES_CSV_PATH": str(csv_path),
                "KILL_SWITCH_PATH": str(tmp_path / "kill.json"),
                "RUNTIME_STATE_PATH": str(tmp_path / "runtime.json"),
                "LIVE_MAX_STEPS": 10,
                "FEATURE_WINDOW_SIZE": 30,
                "MIN_CANDLES": 10,
            }
        ),
    )

    run_paper_session_cli()

    assert "LIVE_MAX_STEPS" in capsys.readouterr().out


def test_register_current_model_creates_registry(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "best_model.joblib"
    model_path.parent.mkdir()
    joblib.dump({"model": DummyEstimator(), "features": ["close"]}, model_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "MODEL_REGISTRY_PATH": str(tmp_path / "models" / "model_registry.json"),
                "PRODUCTION_MODEL_PATH": str(model_path),
            }
        ),
    )

    result = run_register_current_model_cli()

    assert result["status"] == "PRODUCTION"
    assert (tmp_path / "models" / "model_registry.json").exists()


def test_models_lists_registered_model(tmp_path, monkeypatch, capsys):
    registry_path = tmp_path / "registry.json"
    registry = main_module.ModelRegistry(str(registry_path))
    registry.register_model("models/best_model.joblib", {"score": 1.0, "win_rate": 0.6}, "PRODUCTION")
    monkeypatch.setattr(main_module, "settings", main_module.settings.model_copy(update={"MODEL_REGISTRY_PATH": str(registry_path)}))

    models = run_models_cli()

    assert len(models) == 1
    assert "PRODUCTION" in capsys.readouterr().out


def test_summary_current_uses_last_session(tmp_path, monkeypatch):
    repo_path = tmp_path / "trades.db"
    repo = main_module.TradesRepository(str(repo_path))
    repo.insert_trade(
        {
            "session_id": "s1",
            "timestamp": "2026-05-10T12:00:00Z",
            "asset": "EURUSD-OTC",
            "signal": "BUY",
            "confidence": 0.7,
            "amount": 10,
            "entry_price": 100,
            "exit_price": 101,
            "result": "WON",
            "profit": 8.7,
            "balance": 1008.7,
            "reason": "ok",
            "mode": "paper",
        }
    )
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={"TRADES_DB_PATH": str(repo_path), "SESSION_STATE_PATH": str(tmp_path / "session.json")}
        ),
    )

    summary = run_summary_current_cli()

    assert summary["session_id"] == "s1"
    assert summary["net_profit"] == 8.7


def test_sessions_lists_session_metrics(tmp_path, monkeypatch):
    repo_path = tmp_path / "trades.db"
    repo = main_module.TradesRepository(str(repo_path))
    repo.insert_trade(
        {
            "session_id": "s-list",
            "timestamp": "2026-05-10T12:00:00Z",
            "asset": "EURUSD-OTC",
            "signal": "BUY",
            "confidence": 0.7,
            "amount": 10,
            "entry_price": 100,
            "exit_price": 101,
            "result": "WON",
            "profit": 8.7,
            "balance": 1008.7,
            "reason": "ok",
            "mode": "paper",
        }
    )
    manager = main_module.SessionManager(str(tmp_path / "session.json"))
    session = manager.start_session("paper", "EURUSD-OTC", "paper", "mock_realtime")
    manager.update_session(
        session["session_id"],
        {
            "session_id": "s-list",
            "data_start_timestamp": "2026-05-10 12:00:00+00:00",
            "data_end_timestamp": "2026-05-10 12:10:00+00:00",
            "feed_start_index": 10,
            "feed_end_index": 20,
        },
    )
    monkeypatch.setattr(main_module, "settings", main_module.settings.model_copy(update={"TRADES_DB_PATH": str(repo_path), "SESSION_STATE_PATH": str(tmp_path / "session.json")}))

    sessions = run_sessions_cli()
    listed = next(item for item in sessions if item["session_id"] == "s-list")

    assert listed["net_profit"] == 8.7
    assert listed["feed_start_index"] == 10
    assert listed["feed_end_index"] == 20


def test_runtime_status_prints_session_id(tmp_path, monkeypatch, capsys):
    repo_path = tmp_path / "trades.db"
    runtime_path = tmp_path / "runtime.json"
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "TRADES_DB_PATH": str(repo_path),
                "RUNTIME_STATE_PATH": str(runtime_path),
                "SESSION_STATE_PATH": str(tmp_path / "session.json"),
                "KILL_SWITCH_PATH": str(tmp_path / "kill.json"),
            }
        ),
    )
    runtime = main_module.RuntimeState(str(runtime_path))
    runtime.update("last_session_id", "s-runtime")

    result = run_runtime_status_cli()

    assert result["metrics"]["last_session_id"] == "s-runtime"
    assert "last_session_id" in capsys.readouterr().out


def test_feed_status_calculates_remaining_candles(tmp_path, monkeypatch):
    csv_path = tmp_path / "candles.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="min", tz="UTC"),
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
            "volume": [100] * 20,
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "CANDLES_CSV_PATH": str(csv_path),
                "FEED_CURSOR_PATH": str(tmp_path / "cursor.json"),
                "MIN_CANDLES": 10,
                "FEATURE_WINDOW_SIZE": 10,
                "LIVE_MAX_STEPS": 5,
            }
        ),
    )

    status = run_feed_status_cli()

    assert status["total_candles"] == 20
    assert status["remaining_candles"] == 20


def test_feed_status_reports_end_of_feed(tmp_path, monkeypatch):
    csv_path = tmp_path / "candles.csv"
    cursor_path = tmp_path / "feed_cursor.json"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
            "open": [100] * 5,
            "high": [101] * 5,
            "low": [99] * 5,
            "close": [100] * 5,
            "volume": [100] * 5,
        }
    ).to_csv(csv_path, index=False)
    cursor_path.write_text(
        '{"source":"mock_realtime","csv_path":"%s","asset":"EURUSD-OTC","timeframe_seconds":60,"last_index":4}'
        % str(csv_path).replace("\\", "/"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "CANDLES_CSV_PATH": str(csv_path),
                "FEED_CURSOR_PATH": str(cursor_path),
                "MIN_CANDLES": 3,
                "FEATURE_WINDOW_SIZE": 3,
                "LIVE_MAX_STEPS": 5,
            }
        ),
    )

    status = run_feed_status_cli()

    assert status["status"] == "END_OF_FEED"
    assert status["remaining_candles"] == 0


def test_reset_feed_cursor_removes_cursor(tmp_path, monkeypatch):
    cursor = tmp_path / "cursor.json"
    cursor.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main_module, "settings", main_module.settings.model_copy(update={"FEED_CURSOR_PATH": str(cursor)}))

    result = run_reset_feed_cursor_cli(confirm=True)

    assert result["reset"] is True
    assert not cursor.exists()


def test_reset_feed_cursor_requires_confirm(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "settings", main_module.settings.model_copy(update={"FEED_CURSOR_PATH": str(tmp_path / "cursor.json")}))

    with pytest.raises(RuntimeError, match="requires --confirm"):
        run_reset_feed_cursor_cli(confirm=False)


def test_run_paper_session_blocks_duplicate_data_window(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "candles.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=80, freq="min", tz="UTC"),
            "open": [100] * 80,
            "high": [101] * 80,
            "low": [99] * 80,
            "close": [100] * 80,
            "volume": [100] * 80,
        }
    ).to_csv(csv_path, index=False)
    session_path = tmp_path / "current_session.json"
    manager = main_module.SessionManager(str(session_path))
    session = manager.start_session("paper", "EURUSD-OTC", "paper", "mock_realtime")
    manager.update_session(
        session["session_id"],
        {
            "data_start_timestamp": "2026-01-01 00:00:00+00:00",
            "data_end_timestamp": "2026-01-01 00:09:00+00:00",
            "feed_start_index": 0,
            "feed_end_index": 9,
        },
    )
    manager.end_session(session["session_id"], "completed")
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "BOT_MODE": "paper",
                "CANDLES_CSV_PATH": str(csv_path),
                "SESSION_STATE_PATH": str(session_path),
                "RUNTIME_STATE_PATH": str(tmp_path / "runtime.json"),
                "KILL_SWITCH_PATH": str(tmp_path / "kill.json"),
                "FEED_CURSOR_PATH": str(tmp_path / "cursor.json"),
                "LIVE_MAX_STEPS": 10,
                "MIN_CANDLES": 10,
                "FEATURE_WINDOW_SIZE": 10,
                "ALLOW_REPLAY_SAME_WINDOW": False,
            }
        ),
    )

    run_paper_session_cli()

    assert "Duplicate data window blocked" in capsys.readouterr().out


def test_run_paper_session_handles_end_of_feed_without_traceback(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "candles.csv"
    cursor_path = tmp_path / "feed_cursor.json"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="min", tz="UTC"),
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
            "volume": [100] * 20,
        }
    ).to_csv(csv_path, index=False)
    cursor_path.write_text(
        '{"source":"mock_realtime","csv_path":"%s","asset":"EURUSD-OTC","timeframe_seconds":60,"last_index":19}'
        % str(csv_path).replace("\\", "/"),
        encoding="utf-8",
    )
    session_path = tmp_path / "current_session.json"
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(
            update={
                "BOT_MODE": "paper",
                "CANDLES_CSV_PATH": str(csv_path),
                "SESSION_STATE_PATH": str(session_path),
                "RUNTIME_STATE_PATH": str(tmp_path / "runtime.json"),
                "KILL_SWITCH_PATH": str(tmp_path / "kill.json"),
                "FEED_CURSOR_PATH": str(cursor_path),
                "LIVE_MAX_STEPS": 20,
                "MIN_CANDLES": 10,
                "FEATURE_WINDOW_SIZE": 10,
            }
        ),
    )

    run_paper_session_cli()

    assert "No remaining candles to run a new paper session." in capsys.readouterr().out
    assert not session_path.exists()


def test_append_candles_adds_new_candles_and_deduplicates(tmp_path, monkeypatch):
    target = tmp_path / "candles.csv"
    source = tmp_path / "new.csv"
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [100] * 3,
        }
    )
    incoming = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:02:00+00:00", periods=3, freq="min"),
            "open": [102, 103, 104],
            "high": [103, 104, 105],
            "low": [101, 102, 103],
            "close": [102, 103, 104],
            "volume": [100] * 3,
        }
    )
    base.to_csv(target, index=False)
    incoming.to_csv(source, index=False)
    monkeypatch.setattr(
        main_module,
        "settings",
        main_module.settings.model_copy(update={"CANDLES_CSV_PATH": str(target), "ASSET": "EURUSD-OTC", "TIMEFRAME_SECONDS": 60}),
    )

    result = run_append_candles_cli(str(source))

    assert result["new_candles_added"] == 2
    assert len(pd.read_csv(target)) == 5
