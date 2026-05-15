import pandas as pd
import streamlit as st

from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.execution.kill_switch import KillSwitch
from app.mlops.model_registry import ModelRegistry
from app.monitoring.log_watcher import LogWatcher
from app.monitoring.performance_monitor import PerformanceMonitor
from app.monitoring.runtime_metrics import RuntimeMetrics
from app.monitoring.system_monitor import SystemMonitor
from app.runtime.runtime_state import RuntimeState
from app.runtime.session_manager import SessionManager
from app.storage.trades_repository import TradesRepository


settings = get_settings()
repository = TradesRepository(settings.TRADES_DB_PATH)
summary = repository.get_summary()
performance = PerformanceMonitor(repository, settings).get_metrics()
kill = KillSwitch(settings.KILL_SWITCH_PATH)
trades = pd.DataFrame(repository.get_all_trades())
runtime_state = RuntimeState(settings.RUNTIME_STATE_PATH)
session_manager = SessionManager(settings.SESSION_STATE_PATH)
registry = ModelRegistry(settings.MODEL_REGISTRY_PATH)

st.set_page_config(page_title="BOT-YEDA Trading Monitor", layout="wide")
st.title("BOT-YEDA Trading Monitor")

st.caption(f"Mode: {settings.BOT_MODE} | Broker: {settings.BROKER} | Data feed: {settings.DATA_FEED_SOURCE}")
tabs = st.tabs(["Live Trading", "Trades", "Validation", "Models", "Drift", "Logs", "System"])

with tabs[0]:
    if kill.is_active():
        st.error(f"Kill switch ON: {kill.get_reason()}")
    else:
        st.success("Kill switch OFF")
    st.write({"session": session_manager.get_current_session(), "runtime": runtime_state.load()})
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Balance", f"{summary['current_balance']:.2f}")
    col2.metric("Trades", summary["total_trades"])
    col3.metric("Wins", summary["wins"])
    col4.metric("Losses", summary["losses"])
    col5.metric("Win rate", f"{summary['win_rate']:.2%}")
    col6, col7, col8 = st.columns(3)
    col6.metric("Profit factor", f"{performance['profit_factor']:.2f}")
    col7.metric("Daily loss", f"{performance['daily_loss']:.2f}")
    col8.metric("Consecutive losses", performance["max_consecutive_losses"])

with tabs[1]:
    if trades.empty:
        st.info("No live paper/demo trades recorded yet.")
    else:
        if "balance" in trades.columns:
            st.line_chart(trades["balance"])
        st.dataframe(trades, use_container_width=True)

with tabs[2]:
    report_path = Path("data/logs/final_validation_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.json(report)
    else:
        st.info("No validation report yet.")

with tabs[3]:
    models = registry.list_models()
    current = registry.get_current_model()
    st.write({"current_model": current})
    st.dataframe(pd.DataFrame(models), use_container_width=True)

with tabs[4]:
    drift_path = Path("data/logs/model_drift_report.json")
    st.json(json.loads(drift_path.read_text(encoding="utf-8")) if drift_path.exists() else {"status": "not available"})

with tabs[5]:
    st.dataframe(pd.DataFrame(LogWatcher(settings.ALERTS_LOG_PATH).read_recent_alerts()), use_container_width=True)

with tabs[6]:
    st.json({"runtime_metrics": RuntimeMetrics(repository, runtime_state).collect(), "system_metrics": SystemMonitor().collect()})
