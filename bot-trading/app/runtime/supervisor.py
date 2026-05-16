import inspect

from app.execution.kill_switch import KillSwitch


class BotSupervisor:
    def __init__(self, session_manager, runtime_state, healthcheck, alert_manager, settings):
        self.session_manager = session_manager
        self.runtime_state = runtime_state
        self.healthcheck = healthcheck
        self.alert_manager = alert_manager
        self.settings = settings

    def run_with_supervision(self, engine_callable) -> None:
        if self.settings.BOT_MODE == "real":
            raise RuntimeError("Real trading is disabled.")
        session = self.session_manager.start_session(
            self.settings.BOT_MODE,
            self.settings.ASSET,
            self.settings.BROKER,
            self.settings.DATA_FEED_SOURCE,
        )
        self.runtime_state.update("current_session_id", session["session_id"])
        self.runtime_state.update("last_session_id", session["session_id"])
        self.runtime_state.update("bot_status", "RUNNING")
        try:
            KillSwitch(self.settings.KILL_SWITCH_PATH)
            if hasattr(self.healthcheck, "__call__"):
                report = self.healthcheck()
                if report.get("status") == "ERROR":
                    raise RuntimeError("Healthcheck failed.")
            if len(inspect.signature(engine_callable).parameters) >= 1:
                result = engine_callable(session)
            else:
                result = engine_callable()
            if isinstance(result, dict):
                self.session_manager.update_session(session["session_id"], result)
            self.session_manager.save_heartbeat(session["session_id"])
            self.runtime_state.update("bot_status", "STOPPED")
            self.session_manager.end_session(session["session_id"], "completed")
        except Exception as exc:
            self.runtime_state.update("last_error", str(exc))
            self.runtime_state.update("bot_status", "FAILED")
            self.alert_manager.alert("ERROR", f"Supervisor captured error: {exc}")
            if self._is_critical(str(exc)):
                KillSwitch(self.settings.KILL_SWITCH_PATH).activate(str(exc))
            self.session_manager.end_session(session["session_id"], f"failed: {exc}")
            raise

    def _is_critical(self, message: str) -> bool:
        lowered = message.lower()
        return any(token in lowered for token in ["connection", "model", "corrupt", "balance", "real", "kill"])
