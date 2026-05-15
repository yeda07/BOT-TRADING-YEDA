from pathlib import Path

import joblib

from app.execution.execution_state import ExecutionState
from app.execution.kill_switch import KillSwitch
from app.utils.metrics import breakeven_win_rate


class ExecutionGuard:
    def __init__(self, settings, risk_manager, metrics_path: str):
        self.settings = settings
        self.risk_manager = risk_manager
        self.metrics_path = Path(metrics_path)
        self.kill_switch = KillSwitch(settings.KILL_SWITCH_PATH)
        self.execution_state = ExecutionState(settings.EXECUTION_STATE_PATH)

    def validate_before_order(self, signal: str, confidence: float, balance: float, asset: str, timestamp) -> tuple[bool, str]:
        timestamp = str(timestamp)
        if self.kill_switch.is_active():
            return False, f"Kill switch active: {self.kill_switch.get_reason()}"
        if signal == "HOLD":
            return False, "Signal is HOLD."
        if confidence < self.settings.MIN_CONFIDENCE:
            return False, "Confidence below minimum threshold."
        if self.settings.BOT_MODE == "real" or (not self.settings.ENABLE_REAL_TRADING and self.settings.BOT_MODE == "real"):
            return False, "Real trading is disabled."
        if not asset:
            return False, "Asset is required."
        if balance <= 0:
            return False, "Balance must be positive."
        if self.execution_state.was_already_traded(asset, timestamp):
            return False, "This candle was already traded."
        if not self.risk_manager.can_trade():
            return False, "RiskManager does not allow trading."
        ok, reason = self._validate_model_metrics()
        if not ok:
            return False, reason
        return True, "Execution accepted."

    def mark_traded(self, asset: str, timestamp) -> None:
        self.execution_state.set_last_traded_timestamp(asset, str(timestamp))

    def _validate_model_metrics(self) -> tuple[bool, str]:
        if not self.metrics_path.exists():
            return False, f"Model artifact not found: {self.metrics_path}"
        try:
            artifact = joblib.load(self.metrics_path)
        except Exception as exc:
            return False, f"Model metrics could not be loaded: {exc}"
        if not isinstance(artifact, dict):
            return False, "Model artifact does not include metrics."
        metrics = artifact.get("metrics", {})
        test = metrics.get("test", metrics)
        win_rate = float(test.get("win_rate_simulado", 0.0))
        profit_factor = float(test.get("profit_factor_simulado", 0.0))
        drawdown = float(test.get("max_drawdown_simulado", 0.0))
        breakeven = float(artifact.get("breakeven_win_rate", breakeven_win_rate(self.settings.PAYOUT)))
        if win_rate < breakeven:
            return False, "Model win rate does not beat breakeven."
        if profit_factor < self.settings.MIN_PROFIT_FACTOR:
            return False, "Model profit factor is below minimum."
        if drawdown > self.settings.MAX_ALLOWED_DRAWDOWN:
            return False, "Model drawdown is above allowed limit."
        return True, "Model metrics accepted."
