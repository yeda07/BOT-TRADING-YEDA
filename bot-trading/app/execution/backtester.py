from dataclasses import dataclass
from logging import Logger

import pandas as pd

from app.market.features import build_features
from app.risk.risk_manager import RiskManager, RiskState
from app.strategies.rule_based import StrategySignal


TRADE_COLUMNS = [
    "timestamp",
    "strategy",
    "signal",
    "confidence",
    "entry",
    "exit",
    "stake",
    "pnl",
    "balance",
    "reason",
]


@dataclass(frozen=True)
class BacktestMetrics:
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_profit: float
    max_drawdown: float
    profit_factor: float
    final_balance: float


class Backtester:
    def __init__(
        self,
        strategy,
        risk_manager: RiskManager,
        initial_balance: float = 1000.0,
        payout: float = 0.80,
        logger: Logger | None = None,
        strategy_name: str | None = None,
    ) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_balance = initial_balance
        self.payout = payout
        self.logger = logger
        self.strategy_name = strategy_name or strategy.__class__.__name__

    def run(self, candles: pd.DataFrame) -> tuple[BacktestMetrics, pd.DataFrame]:
        featured = build_features(candles)
        state = RiskState(balance=self.initial_balance, starting_balance=self.initial_balance)
        equity_curve = [self.initial_balance]
        trade_rows: list[dict] = []

        for index in range(1, len(featured) - 1):
            history = featured.iloc[: index + 1]
            row = featured.iloc[index]
            signal: StrategySignal = self.strategy.generate_signal(row)
            decision = self.risk_manager.evaluate(signal.signal, signal.confidence, history, state)

            if not decision.allowed:
                continue

            entry = float(row["close"])
            exit_price = float(featured.iloc[index + 1]["close"])
            won = (signal.signal == "BUY" and exit_price > entry) or (signal.signal == "SELL" and exit_price < entry)
            pnl = decision.stake * self.payout if won else -decision.stake
            RiskManager.update_after_trade(state, pnl)
            equity_curve.append(state.balance)

            trade_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "strategy": self.strategy_name,
                    "signal": signal.signal,
                    "confidence": signal.confidence,
                    "entry": entry,
                    "exit": exit_price,
                    "stake": decision.stake,
                    "pnl": pnl,
                    "balance": state.balance,
                    "reason": signal.reason,
                }
            )
            if self.logger:
                self.logger.info(
                    "Trade executed | strategy=%s timestamp=%s signal=%s confidence=%.4f "
                    "entry=%.6f exit=%.6f stake=%.2f pnl=%.2f balance=%.2f reason=%s",
                    self.strategy_name,
                    row["timestamp"],
                    signal.signal,
                    signal.confidence,
                    entry,
                    exit_price,
                    decision.stake,
                    pnl,
                    state.balance,
                    signal.reason,
                )

        trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
        metrics = self._metrics(trades, equity_curve, state.balance)
        return metrics, trades

    def _metrics(self, trades: pd.DataFrame, equity_curve: list[float], final_balance: float) -> BacktestMetrics:
        if trades.empty:
            return BacktestMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, final_balance)

        wins = int((trades["pnl"] > 0).sum())
        losses = int((trades["pnl"] < 0).sum())
        gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        gross_loss = abs(float(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
        equity = pd.Series(equity_curve)
        drawdown = (equity.cummax() - equity) / equity.cummax()
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
        return BacktestMetrics(
            trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=wins / len(trades),
            net_profit=float(trades["pnl"].sum()),
            max_drawdown=float(drawdown.max()),
            profit_factor=float(profit_factor),
            final_balance=final_balance,
        )
