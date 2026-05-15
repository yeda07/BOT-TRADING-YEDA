import json
from datetime import datetime, timezone
from pathlib import Path


class RuntimeMetrics:
    def __init__(self, trades_repository, runtime_state):
        self.trades_repository = trades_repository
        self.runtime_state = runtime_state

    def collect(self) -> dict:
        state = self.runtime_state.load()
        summary = self.trades_repository.get_summary()
        started_at = state.get("started_at")
        uptime = 0.0
        if started_at:
            uptime = (datetime.now(timezone.utc) - datetime.fromisoformat(started_at)).total_seconds()
        metrics = {
            "bot_status": state.get("bot_status"),
            "current_balance": state.get("current_balance") or summary.get("current_balance"),
            "total_trades": summary.get("total_trades"),
            "win_rate": summary.get("win_rate"),
            "profit_factor": _profit_factor(self.trades_repository.get_all_trades()),
            "net_profit": summary.get("net_profit"),
            "max_drawdown": 0.0,
            "last_signal": state.get("last_signal"),
            "last_confidence": state.get("last_confidence"),
            "last_trade_result": state.get("last_trade_result"),
            "last_error": state.get("last_error"),
            "uptime_seconds": uptime,
        }
        path = Path("data/logs/runtime_metrics.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def _profit_factor(trades: list[dict]) -> float:
    profits = [float(trade.get("profit") or 0.0) for trade in trades if trade.get("result") in {"WON", "LOST"}]
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))
    return gross_profit / gross_loss if gross_loss else (999.0 if gross_profit else 0.0)
