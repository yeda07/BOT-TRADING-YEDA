from types import SimpleNamespace

from app.mlops.daily_report import DailyReportBuilder
from app.storage.trades_repository import TradesRepository


def test_daily_report_builder_generates_json_and_markdown(tmp_path):
    repo = TradesRepository(str(tmp_path / "trades.db"))
    settings = SimpleNamespace(
        REPORTS_DIR=str(tmp_path / "reports"),
        BOT_MODE="paper",
        BROKER="paper",
        ASSET="EURUSD-OTC",
        PAYOUT=0.87,
        SESSION_STATE_PATH=str(tmp_path / "session.json"),
        KILL_SWITCH_PATH=str(tmp_path / "kill.json"),
        DRIFT_MIN_TRADES=100,
        DRIFT_WIN_RATE_DROP=0.05,
        MIN_PROFIT_FACTOR=1.0,
        DRIFT_PROFIT_FACTOR_DROP=0.2,
        MIN_CONFIDENCE=0.58,
    )

    report = DailyReportBuilder(repo, settings).build()

    assert report["asset"] == "EURUSD-OTC"
    assert "session_id" in report
    assert "session_profit" in report
    assert list((tmp_path / "reports").glob("daily_report_*.json"))
    assert list((tmp_path / "reports").glob("daily_report_*.md"))
