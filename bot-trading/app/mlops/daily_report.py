import json
from datetime import datetime
from pathlib import Path

from app.execution.kill_switch import KillSwitch
from app.mlops.model_drift_detector import ModelDriftDetector
from app.runtime.session_manager import SessionManager
from app.utils.metrics import breakeven_win_rate, max_consecutive_losses, max_drawdown, profit_factor

import pandas as pd


class DailyReportBuilder:
    def __init__(self, trades_repository, settings):
        self.trades_repository = trades_repository
        self.settings = settings

    def build(self) -> dict:
        trades = self.trades_repository.get_all_trades()
        session = SessionManager(self.settings.SESSION_STATE_PATH).get_current_session()
        session_id = (session or {}).get("session_id") or self.trades_repository.get_last_session_id()
        session_summary = self.trades_repository.get_summary(session_id) if session_id else self.trades_repository.get_summary("__missing__")
        historical_summary = self.trades_repository.get_summary()
        closed = [trade for trade in trades if trade.get("result") in {"WON", "LOST"}]
        profits = pd.Series([float(trade.get("profit") or 0.0) for trade in closed])
        wins = sum(1 for trade in closed if trade.get("result") == "WON")
        losses = sum(1 for trade in closed if trade.get("result") == "LOST")
        balances = pd.Series([float(trade.get("balance") or 0.0) for trade in closed]) if closed else pd.Series(dtype=float)
        confidences = [float(trade.get("confidence") or 0.0) for trade in trades]
        drift = ModelDriftDetector(self.trades_repository, self.settings).analyze()
        kill = KillSwitch(self.settings.KILL_SWITCH_PATH)
        report = {
            "date": datetime.now().strftime("%Y%m%d"),
            "scope": "historical_all_sessions_and_current_session",
            "session_id": session_id,
            "mode": self.settings.BOT_MODE,
            "broker": self.settings.BROKER,
            "asset": self.settings.ASSET,
            "total_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(closed) if closed else 0.0,
            "breakeven_win_rate": breakeven_win_rate(self.settings.PAYOUT),
            "profit_factor": float(profit_factor(profits)) if not profits.empty else 0.0,
            "net_profit": float(profits.sum()) if not profits.empty else 0.0,
            "session_profit": session_summary["net_profit"],
            "historical_profit": historical_summary["net_profit"],
            "session_balance_start": session_summary["initial_balance"],
            "session_balance_end": session_summary["current_balance"],
            "duplicate_data_window_detected": _duplicate_data_window_detected(
                self.trades_repository.get_sessions_summary(),
                SessionManager(self.settings.SESSION_STATE_PATH).list_sessions(),
            ),
            "max_drawdown": float(max_drawdown(balances)) if not balances.empty else 0.0,
            "consecutive_losses": max_consecutive_losses(profits) if not profits.empty else 0,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "best_trade": float(profits.max()) if not profits.empty else 0.0,
            "worst_trade": float(profits.min()) if not profits.empty else 0.0,
            "drift_status": drift,
            "kill_switch": {"active": kill.is_active(), "reason": kill.get_reason()},
            "recommendation": drift["recommendation"],
        }
        self._write(report)
        return report

    def _write(self, report: dict) -> None:
        directory = Path(self.settings.REPORTS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"daily_report_{report['date']}.json"
        md_path = directory / f"daily_report_{report['date']}.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text("# Daily Trading Report\n\n" + "\n".join(f"- {k}: {v}" for k, v in report.items()), encoding="utf-8")


def _duplicate_data_window_detected(repo_sessions: list[dict], history: list[dict]) -> bool:
    seen = set()
    by_id = {session.get("session_id"): dict(session) for session in history}
    for session in repo_sessions:
        merged = by_id.get(session.get("session_id"), {})
        merged.update(session)
        by_id[session.get("session_id")] = merged
    sessions = list(by_id.values())
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
