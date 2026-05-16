from pathlib import Path
import json
import shutil
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.brokers.paper_broker import PaperBroker
from app.brokers.demo_stub_broker import DemoStubBroker
from app.brokers.iqoption_broker import IQOptionBroker
from app.brokers.exnova_broker import ExnovaBroker
from app.config import get_settings
from app.execution.executor import Executor
from app.execution.backtester import Backtester
from app.execution.comparison import compare_rule_vs_ml, format_strategy_comparison
from app.execution.live_engine import LiveTradingEngine
from app.execution.demo_executor import DemoExecutor
from app.execution.execution_guard import ExecutionGuard
from app.execution.kill_switch import KillSwitch
from app.execution.order_manager import OrderManager
from app.execution.order_reconciliation import OrderReconciliation
from app.execution.trade_logger import TradeLogger
from app.market.candles import load_candles_csv, validate_candles
from app.market.candle_storage import CandleStorage
from app.market.candle_schema import normalize_candles
from app.market.data_feed_manager import DataFeedManager
from app.market.data_quality import validate_candles_df
from app.market.features import build_features
from app.market.mock_realtime_feed import EndOfFeedError
from app.ml.predict import MLPredictor, predict_latest
from app.ml.train import train_model
from app.monitoring.healthcheck import print_healthcheck, run_healthcheck
from app.monitoring.alerts import AlertManager
from app.monitoring.runtime_metrics import RuntimeMetrics
from app.monitoring.system_monitor import SystemMonitor
from app.risk.risk_manager import RiskManager, RiskState
from app.runtime.recovery import RecoveryManager
from app.runtime.runtime_state import RuntimeState
from app.runtime.session_manager import SessionManager
from app.runtime.supervisor import BotSupervisor
from app.mlops.daily_report import DailyReportBuilder
from app.mlops.model_drift_detector import ModelDriftDetector
from app.mlops.model_promotion import ModelPromotionManager
from app.mlops.model_registry import ModelRegistry
from app.mlops.retraining_pipeline import RetrainingPipeline
from app.storage.database import save_trades
from app.storage.trades_repository import TradesRepository
from app.strategies.rule_based import RuleBasedStrategy
from app.utils.logger import setup_logger
from app.market.features import create_features
from app.validation.data_leakage_audit import DataLeakageAudit
from app.validation.hyperparameter_search import HyperparameterSearch
from app.validation.model_stability import ModelStabilityAnalyzer
from app.validation.monte_carlo import MonteCarloSimulator
from app.validation.overfitting_detector import OverfittingDetector
from app.validation.stress_testing import StressTester
from app.validation.threshold_optimizer import ThresholdOptimizer
from app.validation.validation_report import ValidationReportBuilder
from app.validation.walk_forward import WalkForwardValidator

import joblib

settings = get_settings()
logger = setup_logger(log_dir=settings.LOG_DIR)
app = FastAPI(title=settings.PROJECT_NAME, version="0.1.0")
DEFAULT_CSV_PATH = "data/raw/candles.csv"


class BacktestRequest(BaseModel):
    csv_path: str
    initial_balance: float | None = None
    save_to_db: bool = True


class TrainRequest(BaseModel):
    csv_path: str
    output_path: str = "models/best_model.joblib"


class CompareRequest(BaseModel):
    csv_path: str
    model_path: str = "models/best_model.joblib"
    initial_balance: float | None = None
    save_to_db: bool = True


def make_risk_manager() -> RiskManager:
    return RiskManager(
        balance=settings.INITIAL_BALANCE,
        risk_per_trade=settings.RISK_PER_TRADE,
        max_consecutive_losses=settings.MAX_CONSECUTIVE_LOSSES,
        max_daily_loss_pct=settings.MAX_DAILY_LOSS,
        min_model_confidence=settings.MIN_CONFIDENCE,
        min_candles=settings.MIN_CANDLES,
        max_volatility_multiplier=settings.MAX_VOLATILITY_MULTIPLIER,
        lateral_market_adx_threshold=settings.LATERAL_MARKET_ADX_THRESHOLD,
    )


@app.get("/health")
def health() -> dict[str, str]:
    settings.validate_trading_mode()
    return {"status": "ok", "mode": settings.BOT_MODE}


@app.post("/backtest")
def backtest(request: BacktestRequest) -> dict:
    try:
        candles = load_candles_csv(request.csv_path)
        validate_candles(candles, min_rows=settings.MIN_CANDLES)
        tester = Backtester(
            strategy=RuleBasedStrategy(),
            risk_manager=make_risk_manager(),
            initial_balance=request.initial_balance or settings.INITIAL_BALANCE,
            payout=settings.PAYOUT,
            expiration_candles=settings.EXPIRATION_CANDLES,
            logger=logger,
            strategy_name="rule_based",
        )
        metrics, trades = tester.run(candles)
        saved = save_trades(trades) if request.save_to_db else 0
        logger.info("Backtest completed for %s with %s trades", request.csv_path, metrics.trades)
        return {"metrics": metrics.__dict__, "saved_trades": saved}
    except Exception as exc:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train(request: TrainRequest) -> dict:
    try:
        candles = load_candles_csv(request.csv_path)
        validate_candles(candles, min_rows=settings.MIN_CANDLES)
        metrics = train_model(
            candles,
            output_path=request.output_path,
            expiration_candles=settings.EXPIRATION_CANDLES,
            payout=settings.PAYOUT,
        )
        logger.info("ML model trained and stored at %s", request.output_path)
        return metrics
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/compare-strategies")
def compare_strategies(request: CompareRequest) -> dict:
    try:
        candles = load_candles_csv(request.csv_path)
        validate_candles(candles, min_rows=settings.MIN_CANDLES)
        comparison, trades = compare_rule_vs_ml(
            candles=candles,
            model_path=request.model_path,
            risk_manager_factory=make_risk_manager,
            initial_balance=request.initial_balance or settings.INITIAL_BALANCE,
            payout=settings.PAYOUT,
            expiration_candles=settings.EXPIRATION_CANDLES,
            logger=logger,
        )
        saved = save_trades(trades) if request.save_to_db else 0
        logger.info(
            "Strategy comparison completed for %s with winner=%s",
            request.csv_path,
            comparison.winner,
        )
        return {"comparison": comparison.to_dict(), "saved_trades": saved}
    except Exception as exc:
        logger.exception("Strategy comparison failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run_backtest_cli(csv_path: str, output_csv: str | None = None) -> None:
    csv_path = require_cli_csv(csv_path)
    candles = load_candles_csv(csv_path)
    validate_candles(candles, min_rows=settings.MIN_CANDLES)
    tester = Backtester(
        RuleBasedStrategy(),
        make_risk_manager(),
        settings.INITIAL_BALANCE,
        payout=settings.PAYOUT,
        expiration_candles=settings.EXPIRATION_CANDLES,
        logger=logger,
        strategy_name="rule_based",
    )
    metrics, trades = tester.run(candles)
    print(pd.Series(metrics.__dict__).to_string())
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output_csv, index=False)


def run_train_cli(csv_path: str, output_path: str = "models/best_model.joblib") -> dict:
    csv_path = require_cli_csv(csv_path)
    candles = load_candles_csv(csv_path)
    validate_candles(candles, min_rows=settings.MIN_CANDLES)
    metrics = train_model(
        candles,
        output_path,
        expiration_candles=settings.EXPIRATION_CANDLES,
        payout=settings.PAYOUT,
        min_confidence=settings.MIN_CONFIDENCE,
    )
    print(pd.Series(metrics).to_string())
    _print_breakeven_summary(metrics)
    return metrics


def run_paper_cli(csv_path: str) -> dict[str, str | float]:
    csv_path = require_cli_csv(csv_path)
    candles = load_candles_csv(csv_path)
    validate_candles(candles, min_rows=settings.MIN_CANDLES)

    broker = PaperBroker(initial_balance=settings.INITIAL_BALANCE, candles=candles)
    broker.connect()
    recent_candles = broker.get_candles(settings.ASSET, settings.TIMEFRAME_SECONDS, len(candles))
    featured = build_features(recent_candles)
    if featured.empty:
        raise ValueError("No usable feature rows for paper simulation.")

    state = RiskState(balance=broker.get_balance(), starting_balance=broker.get_balance())
    executor = Executor(
        broker=broker,
        strategy=RuleBasedStrategy(),
        risk_manager=make_risk_manager(),
        state=state,
    )
    result = executor.execute_latest(featured, settings.ASSET, expiration=settings.EXPIRATION_CANDLES)
    result["balance"] = broker.get_balance()
    result["mode"] = "paper"
    print(pd.Series(result).to_string())
    return result


def run_compare_cli(csv_path: str, model_path: str) -> None:
    csv_path = require_cli_csv(csv_path)
    candles = load_candles_csv(csv_path)
    validate_candles(candles, min_rows=settings.MIN_CANDLES)
    comparison, _ = compare_rule_vs_ml(
        candles=candles,
        model_path=model_path,
        risk_manager_factory=make_risk_manager,
        initial_balance=settings.INITIAL_BALANCE,
        payout=settings.PAYOUT,
        expiration_candles=settings.EXPIRATION_CANDLES,
        logger=logger,
    )
    print(format_strategy_comparison(comparison.rule_based, comparison.ml))
    print(pd.Series(comparison.to_dict()).to_string())


def run_predict_latest_cli(csv_path: str, model_path: str) -> dict[str, float | str]:
    csv_path = require_cli_csv(csv_path)
    if not Path(model_path).exists():
        message = f"Model file not found: {model_path}. Run `python -m app.main train` first."
        logger.error(message)
        raise FileNotFoundError(message)
    candles = load_candles_csv(csv_path)
    validate_candles(candles, min_rows=settings.MIN_CANDLES)
    result = predict_latest(candles, model_path=model_path, min_confidence=settings.MIN_CONFIDENCE)
    print("===== LATEST SIGNAL =====")
    print(f"Asset: {settings.ASSET}")
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Probability Up: {result['probability_up']:.2f}")
    print("Decision: Trade allowed only in paper/demo mode")
    return result


def run_live_paper_cli(
    csv_path: str = DEFAULT_CSV_PATH,
    model_path: str = "models/best_model.joblib",
    runtime_settings=None,
    session_id: str = "manual",
) -> dict:
    if settings.BOT_MODE == "real":
        settings.validate_trading_mode()
    active_settings = runtime_settings or _runtime_settings(csv_path)
    if active_settings.BOT_MODE == "real":
        raise RuntimeError("Real trading is disabled.")
    if active_settings.BOT_MODE not in {"paper", "demo", "backtest"}:
        raise RuntimeError(f"Unsupported live-paper mode: {active_settings.BOT_MODE}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run `python -m app.main train` first.")

    if active_settings.DATA_FEED_SOURCE in {"csv", "mock_realtime"} and not Path(active_settings.CANDLES_CSV_PATH).exists():
        raise FileNotFoundError(
            f"CSV file not found: {active_settings.CANDLES_CSV_PATH}. Expected data/raw/candles.csv for live-paper."
        )

    data_feed = DataFeedManager(active_settings).create_feed()
    predictor = MLPredictor(model_path, require_demo_eligible=False)
    broker = PaperBroker(initial_balance=active_settings.INITIAL_BALANCE)
    broker.connect()
    risk_manager = make_risk_manager()
    order_manager = OrderManager(
        broker=broker,
        risk_manager=risk_manager,
        payout=active_settings.PAYOUT,
        expiration_candles=active_settings.EXPIRATION_CANDLES,
    )
    trade_logger = TradeLogger(active_settings.TRADE_LOG_PATH)
    repository = TradesRepository(active_settings.TRADES_DB_PATH)
    engine = LiveTradingEngine(
        data_feed=data_feed,
        predictor=predictor,
        order_manager=order_manager,
        trade_logger=trade_logger,
        trades_repository=repository,
        settings=active_settings,
        candle_storage=CandleStorage(active_settings.COLLECTED_CANDLES_PATH),
        session_id=session_id,
    )
    return engine.run()


def run_summary_cli(session_id: str | None = None, title: str = "TRADING SUMMARY") -> dict:
    summary = TradesRepository(settings.TRADES_DB_PATH).get_summary(session_id)
    summary["duplicate_data_window_detected"] = _duplicate_data_window_detected()
    print(f"===== {title} =====")
    if session_id:
        print(f"Session ID: {session_id}")
    print(f"Total trades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']}")
    print(f"Losses: {summary['losses']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Net profit: {summary['net_profit']:.2f}")
    print(f"Current balance: {summary['current_balance']:.2f}")
    if summary["duplicate_data_window_detected"]:
        print("WARNING: duplicate data window detected across sessions.")
    return summary


def run_summary_current_cli() -> dict:
    repo = TradesRepository(settings.TRADES_DB_PATH)
    session = SessionManager(settings.SESSION_STATE_PATH).get_current_session()
    session_id = (session or {}).get("session_id") if (session or {}).get("status") == "RUNNING" else None
    session_id = session_id or repo.get_last_session_id()
    if not session_id:
        print("No current or previous session found.")
        return repo.get_summary("__missing__")
    return run_summary_cli(session_id, "CURRENT SESSION SUMMARY")


def run_summary_session_cli(session_id: str) -> dict:
    return run_summary_cli(session_id, "SESSION SUMMARY")


def run_sessions_cli() -> list[dict]:
    repo_sessions = TradesRepository(settings.TRADES_DB_PATH).get_sessions_summary()
    history = SessionManager(settings.SESSION_STATE_PATH).list_sessions()
    sessions = _merge_session_history(repo_sessions, history)
    current = SessionManager(settings.SESSION_STATE_PATH).get_current_session()
    if current and not any(item.get("session_id") == current.get("session_id") for item in sessions):
        sessions.append(
            {
                "session_id": current.get("session_id"),
                "started_at": current.get("started_at"),
                "ended_at": current.get("ended_at"),
                "mode": current.get("mode"),
                "total_trades": 0,
                "net_profit": 0.0,
                "current_balance": 0.0,
            }
        )
    if not sessions:
        print("No sessions found.")
        return []
    columns = [
        "session_id",
        "started_at",
        "ended_at",
        "mode",
        "total_trades",
        "net_profit",
        "current_balance",
        "data_start_timestamp",
        "data_end_timestamp",
        "feed_start_index",
        "feed_end_index",
    ]
    frame = pd.DataFrame(sessions)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    print(frame[columns].to_string(index=False))
    return sessions


def run_reset_paper_logs_cli(confirm: bool = False) -> dict:
    if not confirm:
        raise RuntimeError("reset-paper-logs requires --confirm.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = []
    for path_text in [settings.TRADE_LOG_PATH, settings.TRADES_DB_PATH]:
        path = Path(path_text)
        if path.exists():
            archive = path.with_name(f"{path.stem}_{timestamp}{path.suffix}.bak")
            shutil.move(str(path), str(archive))
            archived.append(str(archive))
    TradeLogger(settings.TRADE_LOG_PATH)
    TradesRepository(settings.TRADES_DB_PATH)
    result = {"archived": archived, "trade_log": settings.TRADE_LOG_PATH, "trades_db": settings.TRADES_DB_PATH}
    print(pd.Series(result).to_string())
    return result


def run_feed_status_cli() -> dict:
    status = get_feed_status(settings)
    print("===== FEED STATUS =====")
    print(f"status: {status['status']}")
    print(f"csv_path: {status['csv_path']}")
    print(f"total candles: {status['total_candles']}")
    print(f"cursor last_index: {status['cursor_last_index']}")
    print(f"cursor last_timestamp: {status['cursor_last_timestamp']}")
    print(f"remaining candles: {status['remaining_candles']}")
    print(f"feature_window_size: {status['feature_window_size']}")
    print(f"estimated usable candles: {status['estimated_usable_candles']}")
    return status


def run_reset_feed_cursor_cli(confirm: bool = False) -> dict:
    if not confirm:
        raise RuntimeError("reset-feed-cursor requires --confirm.")
    path = Path(settings.FEED_CURSOR_PATH)
    if path.exists():
        path.unlink()
    result = {"reset": True, "path": str(path)}
    print(pd.Series(result).to_string())
    return result


def run_append_candles_cli(source_csv: str) -> dict:
    source_path = Path(source_csv)
    target_path = Path(settings.CANDLES_CSV_PATH)
    if not source_path.exists():
        raise FileNotFoundError(f"Source candles CSV not found: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target candles CSV not found: {target_path}")

    existing = normalize_candles(pd.read_csv(target_path), settings.ASSET, settings.TIMEFRAME_SECONDS, "csv")
    incoming = normalize_candles(pd.read_csv(source_path), settings.ASSET, settings.TIMEFRAME_SECONDS, "csv")
    before = len(existing)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "asset", "timeframe_seconds"], keep="first")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    added = len(combined) - before
    target_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target_path, index=False)
    result = {"source_csv": str(source_path), "target_csv": str(target_path), "new_candles_added": int(max(0, added)), "total_candles": len(combined)}
    print(pd.Series(result).to_string())
    return result


def run_collect_data_cli(csv_path: str = DEFAULT_CSV_PATH) -> str:
    runtime_settings = _runtime_settings(csv_path)
    feed = DataFeedManager(runtime_settings).create_feed()
    storage = CandleStorage(runtime_settings.COLLECTED_CANDLES_PATH)
    collected = 0
    print("===== COLLECT DATA =====")
    print(f"Source: {runtime_settings.DATA_FEED_SOURCE}")
    print(f"Asset: {runtime_settings.ASSET}")
    print(f"Timeframe: {runtime_settings.TIMEFRAME_SECONDS}")
    feed.connect()
    max_steps = runtime_settings.LIVE_MAX_STEPS
    try:
        while feed.has_next() and (max_steps is None or collected < max_steps):
            storage.append_candle(feed.get_next_candle())
            collected += 1
    finally:
        feed.disconnect()
    storage.remove_duplicates()
    print(f"Candles collected: {collected}")
    print(f"Saved to: {runtime_settings.COLLECTED_CANDLES_PATH}")
    return runtime_settings.COLLECTED_CANDLES_PATH


def run_data_quality_cli(csv_path: str = DEFAULT_CSV_PATH) -> list[dict]:
    paths = [Path(csv_path), Path(settings.COLLECTED_CANDLES_PATH)]
    reports = []
    print("===== DATA QUALITY REPORT =====")
    for path in paths:
        if not path.exists():
            print(f"File: {path}")
            print("Status: WARNING - file not found")
            reports.append({"file": str(path), "status": "WARNING", "errors": ["file not found"]})
            continue
        data = pd.read_csv(path)
        if not data.empty and "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        valid, errors = validate_candles_df(data)
        duplicates = (
            int(data.duplicated(subset=["timestamp", "asset", "timeframe_seconds", "source"]).sum())
            if {"timestamp", "asset", "timeframe_seconds", "source"}.issubset(data.columns)
            else 0
        )
        missing = int(data.isna().sum().sum())
        invalid_prices = _invalid_price_count(data)
        status = "OK" if valid else "ERROR"
        print(f"File: {path}")
        print(f"Rows: {len(data)}")
        print(f"Duplicated candles: {duplicates}")
        print(f"Missing values: {missing}")
        print(f"Invalid prices: {invalid_prices}")
        print(f"Time gaps: {'yes' if any('gaps' in error for error in errors) else 'no'}")
        print(f"Status: {status}")
        reports.append({"file": str(path), "status": status, "errors": errors})
    return reports


def run_dashboard_cli() -> None:
    print("Run dashboard with: streamlit run app/dashboard/streamlit_app.py")


def make_demo_broker():
    if settings.BROKER == "demo_stub":
        return DemoStubBroker(initial_balance=settings.DEMO_INITIAL_BALANCE)
    if settings.BROKER == "iqoption":
        raise NotImplementedError("Authorized demo execution is not implemented for this broker.")
    if settings.BROKER == "exnova":
        raise NotImplementedError("Authorized demo execution is not implemented for this broker.")
    raise RuntimeError("Demo trading selected, but no authorized demo broker adapter is available.")


def run_demo_cli(model_path: str = "models/best_model.joblib", session_id: str = "manual") -> None:
    if settings.BOT_MODE == "real":
        settings.validate_trading_mode()
        raise RuntimeError("Real trading is disabled.")
    if settings.BOT_MODE != "demo" and settings.BROKER != "demo_stub":
        raise RuntimeError("Demo trading selected, but no authorized demo broker adapter is available.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run `python -m app.main train` first.")

    feed = DataFeedManager(settings).create_feed()
    predictor = MLPredictor(model_path)
    broker = make_demo_broker()
    broker.connect()
    risk_manager = make_risk_manager()
    guard = ExecutionGuard(settings, risk_manager, model_path)
    executor = DemoExecutor(
        broker=broker,
        execution_guard=guard,
        risk_manager=risk_manager,
        trade_logger=TradeLogger(settings.TRADE_LOG_PATH),
        trades_repository=TradesRepository(settings.TRADES_DB_PATH),
        settings=settings,
        session_id=session_id,
    )
    print("===== DEMO TRADING =====")
    steps = 0
    feed.connect()
    try:
        while feed.has_next() and (settings.LIVE_MAX_STEPS is None or steps < settings.LIVE_MAX_STEPS):
            candle = feed.get_next_candle()
            window = feed.get_latest_candles()
            steps += 1
            if len(window) < settings.MIN_CANDLES:
                continue
            featured = build_features(window)
            if featured.empty:
                continue
            prediction = predictor.predict_row(featured.iloc[-1])
            result = executor.execute(str(prediction["signal"]), float(prediction["confidence"]), settings.ASSET, featured.iloc[-1])
            print(pd.Series(result).to_string())
    finally:
        feed.disconnect()
        broker.disconnect()


def run_reconcile_cli() -> dict:
    broker = make_demo_broker()
    broker.connect()
    result = OrderReconciliation(broker, TradesRepository(settings.TRADES_DB_PATH)).reconcile_pending_orders()
    broker.disconnect()
    print(pd.Series(result).to_string())
    return result


def run_healthcheck_cli() -> dict:
    report = run_healthcheck(settings)
    print_healthcheck(report)
    return report


def run_kill_on_cli(reason: str) -> None:
    KillSwitch(settings.KILL_SWITCH_PATH).activate(reason)
    print(f"Kill switch ON: {reason}")


def run_kill_off_cli() -> None:
    KillSwitch(settings.KILL_SWITCH_PATH).deactivate()
    print("Kill switch OFF")


def run_kill_status_cli() -> dict:
    kill = KillSwitch(settings.KILL_SWITCH_PATH)
    result = {"active": kill.is_active(), "reason": kill.get_reason()}
    print(pd.Series(result).to_string())
    return result


def _validation_dataset(csv_path: str = DEFAULT_CSV_PATH):
    candles = load_candles_csv(require_cli_csv(csv_path))
    validate_candles(candles, min_rows=100)
    X, y, features = create_features(candles, settings.EXPIRATION_CANDLES)
    return candles, X, y, features


def run_walk_forward_cli(csv_path: str = DEFAULT_CSV_PATH) -> dict:
    candles = load_candles_csv(require_cli_csv(csv_path))
    result = WalkForwardValidator(
        candles,
        settings,
        min(settings.VALIDATION_TRAIN_WINDOW, max(100, len(candles) // 2)),
        min(settings.VALIDATION_TEST_WINDOW, max(60, len(candles) // 5)),
        min(settings.VALIDATION_STEP_SIZE, max(1, len(candles) // 10)),
    ).run()
    print(pd.Series(result["summary"]).to_string())
    return result


def run_optimize_cli(csv_path: str = DEFAULT_CSV_PATH) -> dict:
    candles, X, y, features = _validation_dataset(csv_path)
    audit = DataLeakageAudit().run(pd.concat([X, y.rename("target")], axis=1), features)
    if audit["has_critical_leakage"]:
        raise RuntimeError("Critical data leakage detected. Optimization stopped.")
    result = HyperparameterSearch(settings).run(X, y, features)
    print(pd.Series({"best_model": result["best_model"], "best_score": result["best_score"]}).to_string())
    return result


def _load_validation_model(X, y, features):
    model_path = Path("models/optimized_model.joblib")
    if model_path.exists():
        artifact = joblib.load(model_path)
        return artifact["model"], artifact.get("features", features)
    fallback = Path("models/best_model.joblib")
    if fallback.exists():
        artifact = joblib.load(fallback)
        return artifact["model"], artifact.get("features", features)
    result = HyperparameterSearch(settings).run(X, y, features)
    artifact = joblib.load("models/optimized_model.joblib")
    return artifact["model"], artifact.get("features", features)


def run_threshold_optimize_cli(csv_path: str = DEFAULT_CSV_PATH) -> dict:
    candles, X, y, features = _validation_dataset(csv_path)
    model, features = _load_validation_model(X, y, features)
    result = ThresholdOptimizer().optimize(model, X[features], y, candles, settings)
    print(pd.Series(result["best"]).to_string())
    return result


def run_monte_carlo_cli() -> dict:
    path = Path("data/logs/ml_backtest_results.csv")
    trades = pd.read_csv(path) if path.exists() else pd.DataFrame({"pnl": [1, -1, 1, 1, -1]})
    result = MonteCarloSimulator(trades, simulations=settings.MONTE_CARLO_SIMULATIONS).run()
    print(pd.Series(result["summary"]).to_string())
    return result


def run_stress_test_cli(csv_path: str = DEFAULT_CSV_PATH) -> dict:
    candles, X, y, features = _validation_dataset(csv_path)
    model, features = _load_validation_model(X, y, features)
    result = StressTester(model, candles, features, settings).run()
    print(pd.Series(result["summary"]).to_string())
    return result


def run_leakage_audit_cli(csv_path: str = DEFAULT_CSV_PATH) -> dict:
    _, X, y, features = _validation_dataset(csv_path)
    result = DataLeakageAudit().run(pd.concat([X, y.rename("target")], axis=1), features)
    print(pd.Series({"status": result["status"], "critical": result["has_critical_leakage"]}).to_string())
    return result


def run_validation_report_cli() -> dict:
    report = ValidationReportBuilder().build()
    print(f"Final recommendation: {report['final_recommendation']}")
    return report


def run_validate_cli(csv_path: str = DEFAULT_CSV_PATH) -> dict:
    leakage = run_leakage_audit_cli(csv_path)
    if leakage["has_critical_leakage"]:
        return run_validation_report_cli()
    walk = run_walk_forward_cli(csv_path)
    optimize = run_optimize_cli(csv_path)
    threshold = run_threshold_optimize_cli(csv_path)
    monte = run_monte_carlo_cli()
    stress = run_stress_test_cli(csv_path)
    overfit = OverfittingDetector().analyze(
        {"accuracy": 0.7, "roc_auc": 0.7},
        {"accuracy": 0.65, "roc_auc": 0.65},
        {"accuracy": 0.62, "roc_auc": 0.62, "total_trades": 100, "win_rate": 0.55},
        walk["summary"],
    )
    stability = ModelStabilityAnalyzer().analyze(walk["folds"], threshold["results"], stress["results"])
    report = run_validation_report_cli()
    return {"leakage": leakage, "walk_forward": walk["summary"], "overfitting": overfit, "stability": stability, "report": report}


def _session_tools(active_settings=None):
    active_settings = active_settings or settings
    state = RuntimeState(active_settings.RUNTIME_STATE_PATH)
    sessions = SessionManager(active_settings.SESSION_STATE_PATH)
    alerts = AlertManager(active_settings.ALERTS_LOG_PATH)
    supervisor = BotSupervisor(sessions, state, lambda: run_healthcheck(active_settings), alerts, active_settings)
    return sessions, state, supervisor


def run_paper_session_cli() -> None:
    if settings.BOT_MODE == "real":
        settings.validate_trading_mode()
        raise RuntimeError("Real trading is disabled.")
    if settings.BOT_MODE != "paper":
        print(f"WARNING: BOT_MODE is {settings.BOT_MODE}. run-paper-session will run with BOT_MODE=paper internally.")
    kill = KillSwitch(settings.KILL_SWITCH_PATH)
    if kill.is_active():
        message = "Kill switch is active. Run: python -m app.main kill-off"
        print(message)
        RuntimeState(settings.RUNTIME_STATE_PATH).save({"bot_status": "BLOCKED", "last_error": message})
        return
    paper_settings = _runtime_settings(settings.CANDLES_CSV_PATH, BOT_MODE="paper", BROKER="paper")
    required_window = max(paper_settings.MIN_CANDLES, paper_settings.FEATURE_WINDOW_SIZE)
    if paper_settings.LIVE_MAX_STEPS is not None and paper_settings.LIVE_MAX_STEPS < required_window:
        message = (
            f"LIVE_MAX_STEPS ({paper_settings.LIVE_MAX_STEPS}) is lower than FEATURE_WINDOW_SIZE/MIN_CANDLES "
            f"({required_window}). Increase LIVE_MAX_STEPS or lower FEATURE_WINDOW_SIZE for a test run."
        )
        print(f"WARNING: {message}")
        RuntimeState(paper_settings.RUNTIME_STATE_PATH).save({"bot_status": "BLOCKED", "last_error": message})
        return
    candle_count = _count_csv_rows(paper_settings.CANDLES_CSV_PATH)
    if candle_count < required_window:
        message = (
            f"Not enough candles for live paper session: {candle_count}/{required_window}. "
            "Run collect-data or provide a larger candles CSV."
        )
        print(message)
        RuntimeState(paper_settings.RUNTIME_STATE_PATH).save({"bot_status": "BLOCKED", "last_error": message})
        return
    feed_status = get_feed_status(paper_settings)
    if feed_status["status"] == "END_OF_FEED":
        message = "No remaining candles to run a new paper session."
        print(message)
        print("Add more candles to data/raw/candles.csv or run reset-feed-cursor --confirm only if you intentionally want replay.")
        RuntimeState(paper_settings.RUNTIME_STATE_PATH).save({"bot_status": "BLOCKED", "last_error": message})
        return
    remaining = int(feed_status["remaining_candles"])
    if remaining < required_window + 50:
        print("WARNING: Not enough remaining candles for a meaningful paper session.")
    if paper_settings.LIVE_MAX_STEPS is not None and paper_settings.LIVE_MAX_STEPS > remaining:
        print(f"WARNING: LIVE_MAX_STEPS ({paper_settings.LIVE_MAX_STEPS}) is greater than remaining candles ({remaining}). Using remaining candles only.")
        paper_settings = paper_settings.model_copy(update={"LIVE_MAX_STEPS": max(1, remaining)})
    planned = _planned_feed_window(paper_settings, feed_status)
    if not paper_settings.ALLOW_REPLAY_SAME_WINDOW and _window_already_used(planned, SessionManager(paper_settings.SESSION_STATE_PATH).list_sessions()):
        message = "Duplicate data window blocked. Set ALLOW_REPLAY_SAME_WINDOW=true only for intentional replay tests."
        print(message)
        RuntimeState(paper_settings.RUNTIME_STATE_PATH).save({"bot_status": "BLOCKED", "last_error": message})
        return
    sessions, state, supervisor = _session_tools(paper_settings)
    state.update("started_at", pd.Timestamp.utcnow().isoformat())
    supervisor.run_with_supervision(
        lambda session: run_live_paper_cli(
            paper_settings.CANDLES_CSV_PATH,
            runtime_settings=paper_settings,
            session_id=session["session_id"],
        )
    )
    run_daily_report_cli()


def run_demo_session_cli() -> None:
    if settings.BOT_MODE != "demo":
        raise RuntimeError("run-demo-session requires BOT_MODE=demo.")
    sessions, state, supervisor = _session_tools()
    state.update("started_at", pd.Timestamp.utcnow().isoformat())
    supervisor.run_with_supervision(lambda session: run_demo_cli(session_id=session["session_id"]))
    run_daily_report_cli()


def run_runtime_status_cli() -> dict:
    sessions = SessionManager(settings.SESSION_STATE_PATH)
    runtime = RuntimeState(settings.RUNTIME_STATE_PATH)
    repo = TradesRepository(settings.TRADES_DB_PATH)
    metrics = RuntimeMetrics(repo, runtime).collect()
    system = SystemMonitor().collect()
    kill = KillSwitch(settings.KILL_SWITCH_PATH)
    result = {
        "session": sessions.get_current_session(),
        "runtime": runtime.load(),
        "kill_switch": {"active": kill.is_active(), "reason": kill.get_reason()},
        "metrics": metrics,
        "system": system,
    }
    print(
        pd.Series(
            {
                "bot_status": metrics["bot_status"],
                "current_session_id": metrics.get("current_session_id"),
                "last_session_id": metrics.get("last_session_id"),
                "kill_switch": result["kill_switch"]["active"],
                "session_trades": metrics.get("session_total_trades"),
                "historical_trades": metrics.get("historical_total_trades"),
                "session_net_profit": metrics.get("session_net_profit"),
                "historical_net_profit": metrics.get("historical_net_profit"),
            }
        ).to_string()
    )
    return result


def run_retrain_cli() -> dict:
    registry = ModelRegistry(settings.MODEL_REGISTRY_PATH)
    result = RetrainingPipeline(settings, registry).run()
    print(pd.Series(result).to_string())
    return result


def run_models_cli() -> list[dict]:
    models = ModelRegistry(settings.MODEL_REGISTRY_PATH).list_models()
    if not models:
        print("No models registered.")
        return models
    rows = []
    for model in models:
        metrics = model.get("metrics", {})
        rows.append(
            {
                "model_id": model.get("model_id"),
                "status": model.get("status"),
                "path": model.get("path"),
                "created_at": model.get("created_at"),
                "score": metrics.get("score"),
                "win_rate": metrics.get("win_rate") or metrics.get("average_win_rate"),
                "profit_factor": metrics.get("profit_factor") or metrics.get("average_profit_factor"),
                "recommendation": metrics.get("final_recommendation"),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    return models


def run_register_current_model_cli() -> dict:
    model_path = Path(settings.PRODUCTION_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"{settings.PRODUCTION_MODEL_PATH} not found. Run `python -m app.main train` first.")
    MLPredictor(model_path, require_demo_eligible=False)
    registry = ModelRegistry(settings.MODEL_REGISTRY_PATH)
    existing = [model for model in registry.list_models() if Path(model.get("path", "")) == model_path]
    if existing:
        print(f"Model already registered: {existing[-1]['model_id']}")
        return existing[-1]
    metrics = _current_model_metrics()
    status = "PRODUCTION" if not registry.get_current_model() else "CANDIDATE"
    model_id = registry.register_model(str(model_path), metrics, status, validation_report_path="data/logs/final_validation_report.json")
    result = registry.get_model(model_id)
    print(pd.Series({"model_id": model_id, "status": status, "path": str(model_path)}).to_string())
    return result


def run_promote_cli(model_id: str) -> dict:
    result = ModelPromotionManager(ModelRegistry(settings.MODEL_REGISTRY_PATH), settings).promote(model_id)
    print(pd.Series(result).to_string())
    return result


def run_rollback_cli(model_id: str) -> dict:
    result = ModelPromotionManager(ModelRegistry(settings.MODEL_REGISTRY_PATH), settings).rollback(model_id)
    print(pd.Series(result).to_string())
    return result


def run_drift_check_cli() -> dict:
    result = ModelDriftDetector(TradesRepository(settings.TRADES_DB_PATH), settings).analyze()
    print(pd.Series({"drift_detected": result["drift_detected"], "risk_level": result["risk_level"], "recommendation": result["recommendation"]}).to_string())
    return result


def run_daily_report_cli() -> dict:
    report = DailyReportBuilder(TradesRepository(settings.TRADES_DB_PATH), settings).build()
    print(pd.Series({"date": report["date"], "total_trades": report["total_trades"], "recommendation": report["recommendation"]}).to_string())
    return report


def _runtime_settings(csv_path: str | None = None, **extra_updates):
    updates = dict(extra_updates)
    if csv_path:
        updates["CANDLES_CSV_PATH"] = csv_path
    return settings.model_copy(update=updates)


def _invalid_price_count(data: pd.DataFrame) -> int:
    price_columns = [column for column in ["open", "high", "low", "close"] if column in data.columns]
    if not price_columns:
        return 0
    prices = data[price_columns].apply(pd.to_numeric, errors="coerce")
    return int((prices <= 0).sum().sum())


def _count_csv_rows(path: str) -> int:
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as file:
        return max(0, sum(1 for _ in file) - 1)


def get_feed_status(active_settings=None) -> dict:
    active_settings = active_settings or settings
    total = _count_csv_rows(active_settings.CANDLES_CSV_PATH)
    feature_window = max(active_settings.MIN_CANDLES, active_settings.FEATURE_WINDOW_SIZE)
    try:
        feed = DataFeedManager(active_settings).create_feed()
        if hasattr(feed, "get_status"):
            status = feed.get_status()
            status["status"] = _feed_status_label(status["remaining_candles"], status["feature_window_size"])
            return status
    except EndOfFeedError:
        cursor = _read_feed_cursor(active_settings.FEED_CURSOR_PATH)
        return {
            "status": "END_OF_FEED",
            "csv_path": active_settings.CANDLES_CSV_PATH,
            "total_candles": total,
            "cursor_last_index": cursor.get("last_index"),
            "cursor_last_timestamp": cursor.get("last_timestamp"),
            "next_start_index": total,
            "remaining_candles": 0,
            "feature_window_size": feature_window,
            "estimated_usable_candles": 0,
        }
    return {
        "status": _feed_status_label(total, feature_window),
        "csv_path": active_settings.CANDLES_CSV_PATH,
        "total_candles": total,
        "cursor_last_index": None,
        "cursor_last_timestamp": None,
        "next_start_index": 0,
        "remaining_candles": total,
        "feature_window_size": feature_window,
        "estimated_usable_candles": max(0, total - feature_window),
    }


def _feed_status_label(remaining: int, feature_window: int) -> str:
    if remaining <= 0:
        return "END_OF_FEED"
    if remaining < feature_window:
        return "NEEDS_MORE_DATA"
    return "OK"


def _read_feed_cursor(path: str) -> dict:
    cursor_path = Path(path)
    if not cursor_path.exists():
        return {}
    try:
        return json.loads(cursor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _planned_feed_window(active_settings, feed_status: dict) -> dict:
    start_index = int(feed_status.get("next_start_index") or 0)
    total = int(feed_status.get("total_candles") or 0)
    steps = active_settings.LIVE_MAX_STEPS or int(feed_status.get("remaining_candles") or 0)
    end_index = min(max(0, total - 1), start_index + max(0, steps) - 1)
    timestamps = _timestamp_range_for_indices(active_settings.CANDLES_CSV_PATH, start_index, end_index)
    return {
        "feed_start_index": start_index,
        "feed_end_index": end_index,
        "data_start_timestamp": timestamps[0],
        "data_end_timestamp": timestamps[1],
    }


def _timestamp_range_for_indices(csv_path: str, start_index: int, end_index: int) -> tuple[str | None, str | None]:
    if end_index < start_index or not Path(csv_path).exists():
        return None, None
    data = pd.read_csv(csv_path, usecols=["timestamp"])
    if data.empty:
        return None, None
    start_index = min(max(0, start_index), len(data) - 1)
    end_index = min(max(0, end_index), len(data) - 1)
    return str(pd.to_datetime(data.iloc[start_index]["timestamp"], errors="coerce")), str(pd.to_datetime(data.iloc[end_index]["timestamp"], errors="coerce"))


def _window_already_used(planned: dict, sessions: list[dict]) -> bool:
    for session in sessions:
        if (
            session.get("data_start_timestamp") == planned.get("data_start_timestamp")
            and session.get("data_end_timestamp") == planned.get("data_end_timestamp")
            and planned.get("data_start_timestamp") is not None
        ):
            return True
    return False


def _duplicate_data_window_detected() -> bool:
    repo_sessions = TradesRepository(settings.TRADES_DB_PATH).get_sessions_summary()
    history = SessionManager(settings.SESSION_STATE_PATH).list_sessions()
    return _has_duplicate_windows(_merge_session_history(repo_sessions, history))


def _has_duplicate_windows(sessions: list[dict]) -> bool:
    seen = set()
    for session in sessions:
        start = session.get("data_start_timestamp") or session.get("started_at")
        end = session.get("data_end_timestamp") or session.get("ended_at")
        if not start or not end:
            continue
        key = (start, end)
        if key in seen:
            return True
        seen.add(key)
    return False


def _merge_session_history(repo_sessions: list[dict], history: list[dict]) -> list[dict]:
    by_id = {session.get("session_id"): dict(session) for session in history}
    for session in repo_sessions:
        session_id = session.get("session_id")
        merged = by_id.get(session_id, {})
        merged.update(session)
        by_id[session_id] = merged
    return list(by_id.values())


def _current_model_metrics() -> dict:
    report_path = Path("data/logs/final_validation_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        wf = report.get("walk_forward_summary", {})
        stability = report.get("model_stability_report", {})
        return {
            "source": "final_validation_report",
            "final_recommendation": report.get("final_recommendation"),
            "average_win_rate": wf.get("average_win_rate"),
            "average_profit_factor": wf.get("average_profit_factor"),
            "stability_score": stability.get("stability_score"),
        }
    metrics_path = Path("data/logs/model_metrics.csv")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if not metrics.empty:
            best = metrics.sort_values("roc_auc", ascending=False).iloc[0].to_dict()
            best["source"] = "model_metrics_csv"
            return best
    return {"source": "model_artifact_only"}


def require_cli_csv(csv_path: str) -> str:
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Expected columns: timestamp,open,high,low,close,volume"
        )
    return csv_path


def _print_breakeven_summary(metrics: dict) -> None:
    best_metrics = metrics["models"][metrics["best_model"]]
    test_metrics = best_metrics.get("test", {})
    win_rate = test_metrics.get("win_rate_simulado", 0.0)
    breakeven = metrics["breakeven_win_rate"]
    print(f"Payout used: {metrics['payout']:.2f}")
    print(f"Minimum breakeven win rate: {breakeven:.2%}")
    print(f"Obtained test win rate: {win_rate:.2%}")
    print(f"Estimated edge: {win_rate - breakeven:.2%}")
    gate = metrics.get("demo_gate", {})
    if gate:
        observed = gate["observed"]
        print(f"Test profit factor: {observed['profit_factor']:.3f}")
        print(f"Test ROC AUC: {observed['roc_auc']:.3f}")
        print(f"Test trades: {observed['total_trades']}")
        print(f"Test max drawdown: {observed['max_drawdown']:.2%}")
    if not metrics.get("eligible_for_demo", False):
        print("DEMO BLOCKED: model does not meet the automatic demo promotion gate.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BOT-YEDA trading bot")
    parser.add_argument(
        "command",
        choices=[
            "backtest",
            "train",
            "paper",
            "compare",
            "predict-latest",
            "live-paper",
            "summary",
            "summary-current",
            "sessions",
            "summary-session",
            "feed-status",
            "append-candles",
            "collect-data",
            "dashboard",
            "data-quality",
            "demo",
            "reconcile",
            "healthcheck",
            "kill-on",
            "kill-off",
            "kill-status",
            "validate",
            "optimize",
            "walk-forward",
            "threshold-optimize",
            "monte-carlo",
            "stress-test",
            "leakage-audit",
            "validation-report",
            "run-paper-session",
            "run-demo-session",
            "runtime-status",
            "retrain",
            "models",
            "register-current-model",
            "promote",
            "rollback",
            "drift-check",
            "daily-report",
            "reset-paper-logs",
            "reset-feed-cursor",
        ],
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to candles CSV")
    parser.add_argument("--output", default=None, help="Output CSV for trades or model path")
    parser.add_argument("--model", default="models/best_model.joblib", help="Model path for strategy comparison")
    parser.add_argument("reason", nargs="?", default="", help="Reason for kill-on")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive maintenance commands")
    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest_cli(args.csv, args.output)
    elif args.command == "train":
        output = args.output or "models/best_model.joblib"
        run_train_cli(args.csv, output)
    elif args.command == "paper":
        run_paper_cli(args.csv)
    elif args.command == "compare":
        run_compare_cli(args.csv, args.model)
    elif args.command == "predict-latest":
        run_predict_latest_cli(args.csv, args.model)
    elif args.command == "live-paper":
        run_live_paper_cli(args.csv, args.model)
    elif args.command == "summary":
        run_summary_cli()
    elif args.command == "summary-current":
        run_summary_current_cli()
    elif args.command == "sessions":
        run_sessions_cli()
    elif args.command == "summary-session":
        run_summary_session_cli(args.reason)
    elif args.command == "feed-status":
        run_feed_status_cli()
    elif args.command == "append-candles":
        run_append_candles_cli(args.reason)
    elif args.command == "collect-data":
        run_collect_data_cli(args.csv)
    elif args.command == "data-quality":
        run_data_quality_cli(args.csv)
    elif args.command == "demo":
        run_demo_cli(args.model)
    elif args.command == "reconcile":
        run_reconcile_cli()
    elif args.command == "healthcheck":
        run_healthcheck_cli()
    elif args.command == "kill-on":
        run_kill_on_cli(args.reason or "manual")
    elif args.command == "kill-off":
        run_kill_off_cli()
    elif args.command == "kill-status":
        run_kill_status_cli()
    elif args.command == "validate":
        run_validate_cli(args.csv)
    elif args.command == "optimize":
        run_optimize_cli(args.csv)
    elif args.command == "walk-forward":
        run_walk_forward_cli(args.csv)
    elif args.command == "threshold-optimize":
        run_threshold_optimize_cli(args.csv)
    elif args.command == "monte-carlo":
        run_monte_carlo_cli()
    elif args.command == "stress-test":
        run_stress_test_cli(args.csv)
    elif args.command == "leakage-audit":
        run_leakage_audit_cli(args.csv)
    elif args.command == "validation-report":
        run_validation_report_cli()
    elif args.command == "run-paper-session":
        run_paper_session_cli()
    elif args.command == "run-demo-session":
        run_demo_session_cli()
    elif args.command == "runtime-status":
        run_runtime_status_cli()
    elif args.command == "retrain":
        run_retrain_cli()
    elif args.command == "models":
        run_models_cli()
    elif args.command == "register-current-model":
        run_register_current_model_cli()
    elif args.command == "promote":
        run_promote_cli(args.reason)
    elif args.command == "rollback":
        run_rollback_cli(args.reason)
    elif args.command == "drift-check":
        run_drift_check_cli()
    elif args.command == "daily-report":
        run_daily_report_cli()
    elif args.command == "reset-paper-logs":
        run_reset_paper_logs_cli(args.confirm)
    elif args.command == "reset-feed-cursor":
        run_reset_feed_cursor_cli(args.confirm)
    else:
        run_dashboard_cli()
