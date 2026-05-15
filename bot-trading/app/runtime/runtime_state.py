import json
from pathlib import Path


DEFAULT_RUNTIME_STATE = {
    "last_candle_timestamp": None,
    "last_signal": None,
    "last_confidence": None,
    "last_trade_result": None,
    "current_balance": 0.0,
    "consecutive_losses": 0,
    "total_trades": 0,
    "last_error": None,
    "bot_status": "IDLE",
}


class RuntimeState:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(DEFAULT_RUNTIME_STATE.copy())

    def save(self, data: dict) -> None:
        payload = DEFAULT_RUNTIME_STATE.copy()
        payload.update(data)
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, key: str, value) -> None:
        data = self.load()
        data[key] = value
        self.save(data)
