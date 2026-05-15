from datetime import datetime, timezone
from pathlib import Path


class AlertManager:
    def __init__(self, path: str = "data/logs/alerts.log"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def alert(self, level: str, message: str) -> None:
        level = level.upper()
        if level not in {"INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Invalid alert level.")
        line = f"{datetime.now(timezone.utc).isoformat()} [{level}] {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
