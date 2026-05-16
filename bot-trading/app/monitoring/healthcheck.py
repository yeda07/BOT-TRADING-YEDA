from pathlib import Path

from app.execution.kill_switch import KillSwitch
from app.mlops.model_registry import ModelRegistry
from app.storage.trades_repository import TradesRepository


def run_healthcheck(settings) -> dict:
    best_model = Path("models/best_model.joblib")
    registry_path = Path(getattr(settings, "MODEL_REGISTRY_PATH", "models/model_registry.json"))
    registry_exists = registry_path.exists()
    production_model = {}
    if registry_exists:
        production_model = ModelRegistry(str(registry_path)).get_current_model()
    candle_count = _count_candles(settings.CANDLES_CSV_PATH)
    warnings = []
    feature_window = int(getattr(settings, "FEATURE_WINDOW_SIZE", getattr(settings, "MIN_CANDLES", 200)))
    if getattr(settings, "LIVE_MAX_STEPS", None) is not None and settings.LIVE_MAX_STEPS < feature_window:
        warnings.append("LIVE_MAX_STEPS is lower than FEATURE_WINDOW_SIZE; paper sessions will finish during warm-up.")
    if settings.BOT_MODE == "backtest":
        warnings.append("BOT_MODE is backtest; run-paper-session will force paper mode but .env should be updated.")
    duplicate_keys = find_duplicate_env_keys(".env")
    if duplicate_keys:
        warnings.append(f"Duplicate keys in .env: {', '.join(duplicate_keys)}")
    checks = {
        "config": settings.BOT_MODE in {"backtest", "paper", "demo", "real"},
        "model": best_model.exists(),
        "model_metrics": Path("data/logs/model_metrics.csv").exists(),
        "data_feed": settings.DATA_FEED_SOURCE in {"csv", "mock_realtime", "iqoption_demo", "exnova_demo"}
        and (Path(settings.CANDLES_CSV_PATH).exists() or settings.DATA_FEED_SOURCE.endswith("_demo")),
        "database": True,
        "real_trading_disabled": not settings.ENABLE_REAL_TRADING,
        "broker": bool(settings.BROKER),
        "model_registry": registry_exists,
        "production_model": bool(production_model),
    }
    TradesRepository(settings.TRADES_DB_PATH)
    kill = KillSwitch(settings.KILL_SWITCH_PATH)
    status = "OK" if all(checks.values()) and not warnings else "WARNING"
    return {
        "checks": checks,
        "bot_mode": settings.BOT_MODE,
        "broker": settings.BROKER,
        "data_feed_source": settings.DATA_FEED_SOURCE,
        "kill_switch": "ON" if kill.is_active() else "OFF",
        "best_model_exists": best_model.exists(),
        "model_registry_exists": registry_exists,
        "production_model_exists": bool(production_model),
        "candle_count": candle_count,
        "warnings": warnings,
        "status": status,
    }


def print_healthcheck(report: dict) -> None:
    print("===== HEALTHCHECK =====")
    print(f"BOT_MODE: {report['bot_mode']}")
    print(f"BROKER: {report['broker']}")
    print(f"DATA_FEED_SOURCE: {report['data_feed_source']}")
    print(f"Config: {'OK' if report['checks']['config'] else 'ERROR'}")
    print(f"best_model.joblib: {'OK' if report['best_model_exists'] else 'MISSING'}")
    print(f"Model registry: {'OK' if report['model_registry_exists'] else 'MISSING'}")
    print(f"Production model: {'OK' if report['production_model_exists'] else 'MISSING'}")
    print(f"Available candles: {report['candle_count']}")
    print(f"Model metrics: {'OK' if report['checks']['model_metrics'] else 'WARNING'}")
    print(f"Data feed: {'OK' if report['checks']['data_feed'] else 'ERROR'}")
    print(f"Database: {'OK' if report['checks']['database'] else 'ERROR'}")
    print(f"Kill switch: {report['kill_switch']}")
    print("Real trading: DISABLED")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Status: {report['status']}")


def find_duplicate_env_keys(path: str) -> list[str]:
    env_path = Path(path)
    if not env_path.exists():
        return []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def _count_candles(path: str) -> int:
    candle_path = Path(path)
    if not candle_path.exists():
        return 0
    try:
        with candle_path.open("r", encoding="utf-8") as file:
            return max(0, sum(1 for _ in file) - 1)
    except OSError:
        return 0
