from types import SimpleNamespace

from app.runtime.recovery import RecoveryManager
from app.runtime.runtime_state import RuntimeState
from app.storage.trades_repository import TradesRepository


def test_recovery_manager_reconstructs_state(tmp_path):
    state = RuntimeState(str(tmp_path / "runtime.json"))
    repo = TradesRepository(str(tmp_path / "trades.db"))
    repo.insert_trade(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "asset": "EURUSD-OTC",
            "signal": "BUY",
            "confidence": 0.7,
            "amount": 10,
            "entry_price": 100,
            "exit_price": 101,
            "result": "WON",
            "profit": 8.7,
            "balance": 1008.7,
            "reason": "test",
            "mode": "paper",
        }
    )

    recovered = RecoveryManager(state, repo, SimpleNamespace()).recover()

    assert recovered["state"]["current_balance"] == 1008.7
    assert recovered["skip_timestamp"] == "2026-01-01T00:00:00Z"
