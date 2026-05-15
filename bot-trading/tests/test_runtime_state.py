from app.runtime.runtime_state import RuntimeState


def test_runtime_state_saves_and_loads(tmp_path):
    state = RuntimeState(str(tmp_path / "runtime.json"))
    state.update("bot_status", "RUNNING")
    state.save({"current_balance": 1234, "bot_status": "STOPPED"})

    data = state.load()

    assert data["current_balance"] == 1234
    assert data["bot_status"] == "STOPPED"
