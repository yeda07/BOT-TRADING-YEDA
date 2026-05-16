import json
from datetime import datetime, timezone
from pathlib import Path


class RuntimeMetrics:
    def __init__(self, trades_repository, runtime_state):
        self.trades_repository = trades_repository
        self.runtime_state = runtime_state

    def collect(self) -> dict:
        state = self.runtime_state.load()
        current_session_id = state.get("current_session_id")
        last_session_id = state.get("last_session_id") or self.trades_repository.get_last_session_id()
        session_id = current_session_id or last_session_id
        historical = self.trades_repository.get_summary()
        session_summary = self.trades_repository.get_summary(session_id) if session_id else self.trades_repository.get_summary("__missing__")
        started_at = state.get("started_at")
        uptime = 0.0
        if started_at:
            uptime = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
        metrics = {
            "bot_status": state.get("bot_status"),
            "current_session_id": current_session_id,
            "last_session_id": last_session_id,
            "current_balance": state.get("current_balance") or session_summary.get("current_balance") or historical.get("current_balance"),
            "total_trades": historical.get("total_trades"),
            "win_rate": historical.get("win_rate"),
            "profit_factor": _profit_factor(self.trades_repository.get_all_trades()),
            "net_profit": historical.get("net_profit"),
            "session_total_trades": session_summary.get("total_trades"),
            "historical_total_trades": historical.get("total_trades"),
            "session_net_profit": session_summary.get("net_profit"),
            "historical_net_profit": historical.get("net_profit"),
            "session_win_rate": session_summary.get("win_rate"),
            "max_drawdown": 0.0,
            "last_signal": state.get("last_signal"),
            "last_confidence": state.get("last_confidence"),
            "last_trade_result": state.get("last_trade_result"),
            "last_error": state.get("last_error"),
            "uptime_seconds": uptime,
        }
        state_path = getattr(self.runtime_state, "path", Path("data/logs/runtime_state.json"))
        path = Path(state_path).parent / "runtime_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def _profit_factor(trades: list[dict]) -> float:
    profits = [float(trade.get("profit") or 0.0) for trade in trades if trade.get("result") in {"WON", "LOST"}]
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))
    return gross_profit / gross_loss if gross_loss else (999.0 if gross_profit else 0.0)
