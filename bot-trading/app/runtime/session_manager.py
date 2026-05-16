import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SessionManager:
    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self.history_path = self.state_path.with_name("sessions.json")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def start_session(self, mode: str, asset: str, broker: str, data_feed: str) -> dict:
        session = {
            "session_id": str(uuid4()),
            "mode": mode,
            "asset": asset,
            "broker": broker,
            "data_feed": data_feed,
            "started_at": _now(),
            "ended_at": None,
            "status": "RUNNING",
            "heartbeat_at": _now(),
            "reason": "",
            "data_start_timestamp": None,
            "data_end_timestamp": None,
            "feed_start_index": None,
            "feed_end_index": None,
            "candle_count_processed": 0,
        }
        self._write(session)
        self._upsert_history(session)
        return session

    def end_session(self, session_id: str, reason: str) -> None:
        session = self.get_current_session() or {}
        if session.get("session_id") != session_id:
            return
        session.update({"ended_at": _now(), "status": "KILLED" if "kill" in reason.lower() else "STOPPED", "reason": reason})
        self._write(session)
        self._upsert_history(session)

    def update_session(self, session_id: str, updates: dict) -> None:
        session = self.get_current_session() or {}
        if session.get("session_id") != session_id:
            return
        session.update(updates)
        self._write(session)
        self._upsert_history(session)

    def list_sessions(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def get_current_session(self) -> dict | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_heartbeat(self, session_id: str) -> None:
        session = self.get_current_session()
        if session and session.get("session_id") == session_id:
            session["heartbeat_at"] = _now()
            self._write(session)

    def is_session_active(self) -> bool:
        session = self.get_current_session()
        return bool(session and session.get("status") == "RUNNING")

    def _write(self, session: dict) -> None:
        self.state_path.write_text(json.dumps(session, indent=2), encoding="utf-8")

    def _upsert_history(self, session: dict) -> None:
        sessions = self.list_sessions()
        sessions = [item for item in sessions if item.get("session_id") != session.get("session_id")]
        sessions.append(session)
        self.history_path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
