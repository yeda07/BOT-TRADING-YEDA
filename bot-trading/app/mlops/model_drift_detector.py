import json
from pathlib import Path

from app.utils.metrics import breakeven_win_rate


class ModelDriftDetector:
    def __init__(self, trades_repository, settings):
        self.trades_repository = trades_repository
        self.settings = settings

    def analyze(self) -> dict:
        trades = [trade for trade in self.trades_repository.get_all_trades() if trade.get("result") in {"WON", "LOST", "HOLD"}]
        closed = [trade for trade in trades if trade.get("result") in {"WON", "LOST"}]
        warnings = []
        if len(closed) < self.settings.DRIFT_MIN_TRADES:
            result = {"drift_detected": False, "risk_level": "LOW", "warnings": ["Not enough trades for drift analysis."], "recommendation": "Keep collecting paper/demo data."}
            self._save(result)
            return result
        wins = sum(1 for trade in closed if trade.get("result") == "WON")
        win_rate = wins / len(closed)
        profits = [float(trade.get("profit") or 0.0) for trade in closed]
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss else 999.0
        hold_ratio = sum(1 for trade in trades if trade.get("result") == "HOLD") / len(trades)
        avg_conf = sum(float(trade.get("confidence") or 0.0) for trade in trades) / len(trades)
        if win_rate < breakeven_win_rate(self.settings.PAYOUT) - self.settings.DRIFT_WIN_RATE_DROP:
            warnings.append("Win rate dropped below acceptable edge.")
        if profit_factor < max(0.1, self.settings.MIN_PROFIT_FACTOR - self.settings.DRIFT_PROFIT_FACTOR_DROP):
            warnings.append("Profit factor deteriorated.")
        if hold_ratio > 0.75:
            warnings.append("HOLD signal ratio increased.")
        if avg_conf < self.settings.MIN_CONFIDENCE:
            warnings.append("Average confidence dropped.")
        risk = "HIGH" if len(warnings) >= 3 else "MEDIUM" if warnings else "LOW"
        result = {"drift_detected": bool(warnings), "risk_level": risk, "warnings": warnings, "recommendation": "Retrain and validate candidate." if warnings else "Continue monitoring."}
        self._save(result)
        return result

    def _save(self, result: dict) -> None:
        path = Path("data/logs/model_drift_report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
