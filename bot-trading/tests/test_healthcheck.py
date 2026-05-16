from types import SimpleNamespace

from app.monitoring.healthcheck import run_healthcheck


def test_healthcheck_returns_ok_with_valid_configuration(tmp_path):
    candles = tmp_path / "candles.csv"
    candles.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    settings = SimpleNamespace(
        BOT_MODE="paper",
        DATA_FEED_SOURCE="mock_realtime",
        CANDLES_CSV_PATH=str(candles),
        TRADES_DB_PATH=str(tmp_path / "trades.db"),
        KILL_SWITCH_PATH=str(tmp_path / "kill.json"),
        ENABLE_REAL_TRADING=False,
        BROKER="demo_stub",
        MODEL_REGISTRY_PATH=str(tmp_path / "registry.json"),
        LIVE_MAX_STEPS=200,
        FEATURE_WINDOW_SIZE=100,
    )

    report = run_healthcheck(settings)

    assert report["checks"]["config"]
    assert report["checks"]["data_feed"]
    assert report["kill_switch"] == "OFF"


def test_healthcheck_reports_useful_warnings(tmp_path):
    candles = tmp_path / "candles.csv"
    candles.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    settings = SimpleNamespace(
        BOT_MODE="backtest",
        DATA_FEED_SOURCE="mock_realtime",
        CANDLES_CSV_PATH=str(candles),
        TRADES_DB_PATH=str(tmp_path / "trades.db"),
        KILL_SWITCH_PATH=str(tmp_path / "kill.json"),
        ENABLE_REAL_TRADING=False,
        BROKER="paper",
        MODEL_REGISTRY_PATH=str(tmp_path / "registry.json"),
        LIVE_MAX_STEPS=10,
        FEATURE_WINDOW_SIZE=300,
    )

    report = run_healthcheck(settings)

    assert report["status"] == "WARNING"
    assert any("LIVE_MAX_STEPS" in warning for warning in report["warnings"])
    assert report["bot_mode"] == "backtest"
