from app.execution.kill_switch import KillSwitch


def test_kill_switch_activates_and_deactivates(tmp_path):
    kill = KillSwitch(str(tmp_path / "kill.json"))

    kill.activate("risk")
    assert kill.is_active()
    assert kill.get_reason() == "risk"

    kill.deactivate()
    assert not kill.is_active()
