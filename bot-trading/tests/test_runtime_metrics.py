from app.monitoring.runtime_metrics import RuntimeMetrics
from app.runtime.runtime_state import RuntimeState
from app.storage.trades_repository import TradesRepository


def test_runtime_metrics_generates_snapshot(tmp_path):
    state = RuntimeState(str(tmp_path / "runtime.json"))
    repo = TradesRepository(str(tmp_path / "trades.db"))
    state.update("bot_status", "RUNNING")
    state.update("last_session_id", "s1")
    repo.insert_trade(
        {
            "session_id": "s1",
            "timestamp": "2026-05-10T12:00:00Z",
            "asset": "EURUSD-OTC",
            "signal": "BUY",
            "confidence": 0.7,
            "amount": 10,
            "entry_price": 100,
            "exit_price": 101,
            "result": "WON",
            "profit": 8.7,
            "balance": 1008.7,
            "reason": "ok",
            "mode": "paper",
        }
    )

    metrics = RuntimeMetrics(repo, state).collect()

    assert metrics["bot_status"] == "RUNNING"
    assert "total_trades" in metrics
    assert metrics["last_session_id"] == "s1"
    assert metrics["session_net_profit"] == 8.7
    assert (tmp_path / "runtime_metrics.json").exists()
