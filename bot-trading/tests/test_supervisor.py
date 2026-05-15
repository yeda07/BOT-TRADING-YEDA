from types import SimpleNamespace

import pytest

from app.monitoring.alerts import AlertManager
from app.runtime.runtime_state import RuntimeState
from app.runtime.session_manager import SessionManager
from app.runtime.supervisor import BotSupervisor


def test_supervisor_captures_error_and_alerts(tmp_path):
    settings = SimpleNamespace(
        BOT_MODE="paper",
        ASSET="EURUSD-OTC",
        BROKER="paper",
        DATA_FEED_SOURCE="mock_realtime",
        KILL_SWITCH_PATH=str(tmp_path / "kill.json"),
    )
    state = RuntimeState(str(tmp_path / "runtime.json"))
    supervisor = BotSupervisor(
        SessionManager(str(tmp_path / "session.json")),
        state,
        lambda: {"status": "OK"},
        AlertManager(str(tmp_path / "alerts.log")),
        settings,
    )

    with pytest.raises(RuntimeError, match="model missing"):
        supervisor.run_with_supervision(lambda: (_ for _ in ()).throw(RuntimeError("model missing")))

    assert state.load()["bot_status"] == "FAILED"
    assert "model missing" in (tmp_path / "alerts.log").read_text(encoding="utf-8")
