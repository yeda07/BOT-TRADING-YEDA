import pandas as pd
import streamlit as st

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.storage.trades_repository import TradesRepository


settings = get_settings()
repository = TradesRepository(settings.TRADES_DB_PATH)
summary = repository.get_summary()
trades = pd.DataFrame(repository.get_all_trades())

st.set_page_config(page_title="BOT-YEDA Paper Trading", layout="wide")
st.title("BOT-YEDA Paper Trading")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Balance", f"{summary['current_balance']:.2f}")
col2.metric("Trades", summary["total_trades"])
col3.metric("Wins", summary["wins"])
col4.metric("Losses", summary["losses"])
col5.metric("Win rate", f"{summary['win_rate']:.2%}")

st.metric("Net profit", f"{summary['net_profit']:.2f}")

if trades.empty:
    st.info("No live paper trades recorded yet.")
else:
    if "balance" in trades.columns:
        st.line_chart(trades["balance"])
    st.dataframe(trades, use_container_width=True)
