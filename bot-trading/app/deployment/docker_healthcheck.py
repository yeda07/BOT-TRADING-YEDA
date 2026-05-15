from app.config import get_settings
from app.monitoring.healthcheck import run_healthcheck


def main() -> int:
    report = run_healthcheck(get_settings())
    return 0 if report["status"] in {"OK", "WARNING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
