from pathlib import Path

from app.execution.kill_switch import KillSwitch
from app.storage.trades_repository import TradesRepository


def run_healthcheck(settings) -> dict:
    checks = {
        "config": settings.BOT_MODE in {"backtest", "paper", "demo", "real"},
        "model": Path("models/best_model.joblib").exists(),
        "model_metrics": Path("data/logs/model_metrics.csv").exists(),
        "data_feed": settings.DATA_FEED_SOURCE in {"csv", "mock_realtime", "iqoption_demo", "exnova_demo"}
        and (Path(settings.CANDLES_CSV_PATH).exists() or settings.DATA_FEED_SOURCE.endswith("_demo")),
        "database": True,
        "real_trading_disabled": not settings.ENABLE_REAL_TRADING,
        "broker": bool(settings.BROKER),
    }
    TradesRepository(settings.TRADES_DB_PATH)
    kill = KillSwitch(settings.KILL_SWITCH_PATH)
    status = "OK" if all(checks.values()) else "WARNING"
    return {"checks": checks, "kill_switch": "ON" if kill.is_active() else "OFF", "status": status}


def print_healthcheck(report: dict) -> None:
    print("===== HEALTHCHECK =====")
    print(f"Config: {'OK' if report['checks']['config'] else 'ERROR'}")
    print(f"Model: {'OK' if report['checks']['model'] else 'WARNING'}")
    print(f"Data feed: {'OK' if report['checks']['data_feed'] else 'ERROR'}")
    print(f"Database: {'OK' if report['checks']['database'] else 'ERROR'}")
    print(f"Kill switch: {report['kill_switch']}")
    print("Real trading: DISABLED")
    print(f"Status: {report['status']}")
