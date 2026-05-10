from dataclasses import dataclass
from logging import Logger
from pathlib import Path

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
    "won",
    "stake",
    "pnl",
    "balance",
    "reason",
]


@dataclass(frozen=True)
class BacktestMetrics:
    initial_balance: float
    final_balance: float
    net_profit: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_consecutive_losses: int
    equity_curve: list[float]

    @property
    def trades(self) -> int:
        return self.total_trades


class Backtester:
    def __init__(
        self,
        strategy,
        risk_manager: RiskManager,
        initial_balance: float = 1000.0,
        payout: float = 0.80,
        stake: float | None = None,
        expiration_candles: int = 1,
        output_dir: str | Path = "data/logs",
        logger: Logger | None = None,
        strategy_name: str | None = None,
    ) -> None:
        if expiration_candles < 1:
            raise ValueError("expiration_candles must be at least 1.")
        if stake is not None and stake <= 0:
            raise ValueError("stake must be positive.")
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_balance = initial_balance
        self.payout = payout
        self.stake = stake
        self.expiration_candles = expiration_candles
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.strategy_name = strategy_name or strategy.__class__.__name__

    def run(self, candles: pd.DataFrame) -> tuple[BacktestMetrics, pd.DataFrame]:
        featured = build_features(candles)
        state = RiskState(balance=self.initial_balance, starting_balance=self.initial_balance)
        equity_rows = [{"timestamp": None, "balance": self.initial_balance, "drawdown": 0.0}]
        trade_rows: list[dict] = []

        for index in range(1, len(featured) - self.expiration_candles):
            history = featured.iloc[: index + 1]
            row = featured.iloc[index]
            signal: StrategySignal = self.strategy.generate_signal(row)
            decision = self.risk_manager.evaluate(signal.signal, signal.confidence, history, state)

            if not decision.allowed:
                continue

            entry = float(row["close"])
            exit_row = featured.iloc[index + self.expiration_candles]
            exit_price = float(exit_row["close"])
            won = (signal.signal == "BUY" and exit_price > entry) or (signal.signal == "SELL" and exit_price < entry)
            stake = self.stake if self.stake is not None else decision.stake
            pnl = stake * self.payout if won else -stake
            RiskManager.update_after_trade(state, pnl)

            trade_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "strategy": self.strategy_name,
                    "signal": signal.signal,
                    "confidence": signal.confidence,
                    "entry": entry,
                    "exit": exit_price,
                    "won": won,
                    "stake": stake,
                    "pnl": pnl,
                    "balance": state.balance,
                    "reason": signal.reason,
                }
            )
            peak_balance = max([item["balance"] for item in equity_rows] + [state.balance])
            drawdown = (peak_balance - state.balance) / peak_balance if peak_balance else 0.0
            equity_rows.append({"timestamp": exit_row["timestamp"], "balance": state.balance, "drawdown": drawdown})
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
                    stake,
                    pnl,
                    state.balance,
                    signal.reason,
                )

        trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
        equity_curve = pd.DataFrame(equity_rows)
        metrics = self._metrics(trades, equity_curve, state.balance)
        self._save_results(trades, equity_curve)
        return metrics, trades

    def _metrics(self, trades: pd.DataFrame, equity_curve: pd.DataFrame, final_balance: float) -> BacktestMetrics:
        equity_values = [float(value) for value in equity_curve["balance"].tolist()]
        if trades.empty:
            return BacktestMetrics(
                initial_balance=self.initial_balance,
                final_balance=final_balance,
                net_profit=0.0,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_consecutive_losses=0,
                equity_curve=equity_values,
            )

        wins = int((trades["pnl"] > 0).sum())
        losses = int((trades["pnl"] < 0).sum())
        gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
        gross_loss = abs(float(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
        equity = pd.Series(equity_values)
        drawdown = (equity.cummax() - equity) / equity.cummax()
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
        return BacktestMetrics(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            net_profit=float(trades["pnl"].sum()),
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=wins / len(trades),
            profit_factor=float(profit_factor),
            max_drawdown=float(drawdown.max()),
            max_consecutive_losses=self._max_consecutive_losses(trades),
            equity_curve=equity_values,
        )

    def _save_results(self, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(self.output_dir / "backtest_results.csv", index=False)
        equity_curve.to_csv(self.output_dir / "equity_curve.csv", index=False)

    @staticmethod
    def _max_consecutive_losses(trades: pd.DataFrame) -> int:
        max_losses = 0
        current_losses = 0
        for pnl in trades["pnl"]:
            if pnl < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        return max_losses
