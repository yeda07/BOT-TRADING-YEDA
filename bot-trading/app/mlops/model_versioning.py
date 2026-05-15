import shutil
from datetime import datetime
from pathlib import Path


def save_versioned_model(model_artifact, versions_dir: str = "models/versions") -> str:
    Path(versions_dir).mkdir(parents=True, exist_ok=True)
    path = Path(versions_dir) / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    if isinstance(model_artifact, (str, Path)):
        shutil.copy2(model_artifact, path)
    else:
        import joblib

        joblib.dump(model_artifact, path)
    return str(path)


def copy_to_production(model_path: str, production_path: str = "models/best_model.joblib") -> None:
    production = Path(production_path)
    production.parent.mkdir(parents=True, exist_ok=True)
    if production.exists():
        backup = production.with_suffix(".backup.joblib")
        shutil.copy2(production, backup)
    shutil.copy2(model_path, production)


def rollback_model(model_path: str, production_path: str = "models/best_model.joblib") -> None:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Rollback model not found: {model_path}")
    copy_to_production(model_path, production_path)
