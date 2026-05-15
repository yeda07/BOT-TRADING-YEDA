from app.runtime.session_manager import SessionManager


def test_session_manager_starts_and_ends_session(tmp_path):
    manager = SessionManager(str(tmp_path / "session.json"))
    session = manager.start_session("paper", "EURUSD-OTC", "paper", "mock_realtime")

    assert manager.is_session_active()
    manager.save_heartbeat(session["session_id"])
    manager.end_session(session["session_id"], "done")

    current = manager.get_current_session()
    assert current["status"] == "STOPPED"
    assert current["ended_at"] is not None
