import json
from pathlib import Path


class KillSwitch:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"active": False, "reason": ""})

    def activate(self, reason: str) -> None:
        self._write({"active": True, "reason": reason})

    def deactivate(self) -> None:
        self._write({"active": False, "reason": ""})

    def is_active(self) -> bool:
        return bool(self._read().get("active", False))

    def get_reason(self) -> str:
        return str(self._read().get("reason", ""))

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
