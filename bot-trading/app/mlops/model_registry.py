import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ModelRegistry:
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"models": []})

    def register_model(self, model_path: str, metrics: dict, status: str, validation_report_path: str = "", notes: str = "") -> str:
        model_id = str(uuid4())
        data = self._read()
        data["models"].append(
            {
                "model_id": model_id,
                "path": model_path,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "status": status,
                "validation_report_path": validation_report_path,
                "notes": notes,
            }
        )
        self._write(data)
        return model_id

    def get_current_model(self) -> dict:
        production = [model for model in self.list_models() if model["status"] == "PRODUCTION"]
        return production[-1] if production else {}

    def get_model(self, model_id: str) -> dict:
        for model in self.list_models():
            if model["model_id"] == model_id:
                return model
        raise KeyError(f"Model not found: {model_id}")

    def list_models(self) -> list[dict]:
        return self._read()["models"]

    def set_status(self, model_id: str, status: str) -> None:
        data = self._read()
        for model in data["models"]:
            if model["model_id"] == model_id:
                model["status"] = status
                self._write(data)
                return
        raise KeyError(f"Model not found: {model_id}")

    def _read(self) -> dict:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
