from app.monitoring.runtime_metrics import RuntimeMetrics
from app.runtime.runtime_state import RuntimeState
from app.storage.trades_repository import TradesRepository


def test_runtime_metrics_generates_snapshot(tmp_path):
    state = RuntimeState(str(tmp_path / "runtime.json"))
    repo = TradesRepository(str(tmp_path / "trades.db"))
    state.update("bot_status", "RUNNING")

    metrics = RuntimeMetrics(repo, state).collect()

    assert metrics["bot_status"] == "RUNNING"
    assert "total_trades" in metrics
