import json
from pathlib import Path


class ExecutionState:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def get_last_traded_timestamp(self, asset: str) -> str | None:
        return self._read().get(asset)

    def set_last_traded_timestamp(self, asset: str, timestamp: str) -> None:
        data = self._read()
        data[asset] = timestamp
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def was_already_traded(self, asset: str, timestamp: str) -> bool:
        return self.get_last_traded_timestamp(asset) == timestamp

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))
