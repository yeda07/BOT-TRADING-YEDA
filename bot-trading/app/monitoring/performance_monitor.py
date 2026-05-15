import math

from app.utils.metrics import breakeven_win_rate


class PerformanceMonitor:
    def __init__(self, trades_repository, settings):
        self.trades_repository = trades_repository
        self.settings = settings

    def get_metrics(self) -> dict:
        trades = [trade for trade in self.trades_repository.get_all_trades() if trade.get("result") in {"WON", "LOST"}]
        profits = [float(trade.get("profit") or 0.0) for trade in trades]
        wins = sum(1 for trade in trades if trade.get("result") == "WON")
        losses = sum(1 for trade in trades if trade.get("result") == "LOST")
        gross_profit = sum(profit for profit in profits if profit > 0)
        gross_loss = abs(sum(profit for profit in profits if profit < 0))
        consecutive = 0
        for profit in reversed(profits):
            if profit < 0:
                consecutive += 1
            else:
                break
        balances = [float(trade.get("balance") or 0.0) for trade in trades if trade.get("balance") is not None]
        return {
            "total_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "net_profit": sum(profits),
            "daily_loss": abs(sum(profit for profit in profits if profit < 0)),
            "max_consecutive_losses": consecutive,
            "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else 0.0),
            "current_balance": balances[-1] if balances else 0.0,
            "wins": wins,
            "losses": losses,
        }

    def should_stop_trading(self) -> tuple[bool, str]:
        metrics = self.get_metrics()
        if metrics["current_balance"] and metrics["daily_loss"] >= metrics["current_balance"] * self.settings.MAX_DAILY_LOSS:
            return True, "Daily loss limit reached."
        if metrics["max_consecutive_losses"] >= self.settings.MAX_CONSECUTIVE_LOSSES:
            return True, "Maximum consecutive losses reached."
        if metrics["total_trades"] > 0 and metrics["profit_factor"] < self.settings.MIN_PROFIT_FACTOR:
            return True, "Profit factor below minimum."
        if metrics["total_trades"] >= self.settings.MIN_TRADES_BEFORE_LIVE and metrics["win_rate"] < breakeven_win_rate(self.settings.PAYOUT):
            return True, "Win rate below breakeven."
        return False, "Performance within limits."
