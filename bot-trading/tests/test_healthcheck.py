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
    )

    report = run_healthcheck(settings)

    assert report["checks"]["config"]
    assert report["checks"]["data_feed"]
    assert report["kill_switch"] == "OFF"
