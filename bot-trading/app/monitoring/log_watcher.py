from pathlib import Path


class LogWatcher:
    def __init__(self, alerts_log_path: str):
        self.alerts_log_path = Path(alerts_log_path)

    def read_recent_alerts(self, limit: int = 50) -> list[dict]:
        if not self.alerts_log_path.exists():
            return []
        lines = self.alerts_log_path.read_text(encoding="utf-8").splitlines()[-limit:]
        alerts = []
        for line in lines:
            parts = line.split(" ", 1)
            alerts.append({"timestamp": parts[0], "message": parts[1] if len(parts) > 1 else line})
        return alerts
