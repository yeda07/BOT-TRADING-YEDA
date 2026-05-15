import json
import shutil
import time
from pathlib import Path


class SystemMonitor:
    def __init__(self):
        self.started_at = time.time()

    def collect(self) -> dict:
        metrics = {"process_uptime_seconds": time.time() - self.started_at}
        try:
            import psutil

            metrics.update({"cpu_percent": psutil.cpu_percent(interval=0), "ram_percent": psutil.virtual_memory().percent})
        except Exception:
            metrics.update({"cpu_percent": None, "ram_percent": None})
        usage = shutil.disk_usage(".")
        metrics["disk_usage_percent"] = usage.used / usage.total * 100
        path = Path("data/logs/system_metrics.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
