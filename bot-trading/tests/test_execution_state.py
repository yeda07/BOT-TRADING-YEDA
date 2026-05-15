from app.execution.execution_state import ExecutionState


def test_execution_state_prevents_duplicate_candle(tmp_path):
    state = ExecutionState(str(tmp_path / "execution_state.json"))
    state.set_last_traded_timestamp("EURUSD-OTC", "2026-01-01T00:00:00Z")

    assert state.was_already_traded("EURUSD-OTC", "2026-01-01T00:00:00Z")
    assert not state.was_already_traded("EURUSD-OTC", "2026-01-01T00:01:00Z")
