from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from app.market.features import build_features
from app.utils.metrics import max_consecutive_losses, max_drawdown, profit_factor


ML_TRADE_COLUMNS = [
    "timestamp",
    "signal",
    "confidence",
    "probability_up",
    "entry",
    "exit",
    "won",
    "stake",
    "pnl",
    "balance",
]


@dataclass(frozen=True)
class MLBacktestMetrics:
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
    average_profit_per_trade: float
    total_hold_signals: int

    def to_dict(self) -> dict:
        return asdict(self)


class MLBacktester:
    def __init__(
        self,
        model,
        features: list[str],
        payout: float = 0.87,
        stake: float = 10.0,
        initial_balance: float = 1000.0,
        expiration_candles: int = 1,
        min_confidence: float = 0.58,
        output_dir: str | Path = "data/logs",
    ) -> None:
        if expiration_candles < 1:
            raise ValueError("expiration_candles must be at least 1.")
        if stake <= 0:
            raise ValueError("stake must be positive.")
        if not features:
            raise ValueError("features must not be empty.")
        self.model = model
        self.features = features
        self.payout = payout
        self.stake = stake
        self.initial_balance = initial_balance
        self.expiration_candles = expiration_candles
        self.min_confidence = min_confidence
        self.output_dir = Path(output_dir)

    def run(self, df: pd.DataFrame) -> tuple[MLBacktestMetrics, pd.DataFrame]:
        featured = df.copy()
        if not set(self.features).issubset(featured.columns):
            featured = build_features(featured)
        missing = set(self.features) - set(featured.columns)
        if missing:
            raise ValueError(f"ML backtest missing features: {sorted(missing)}")

        balance = self.initial_balance
        trades: list[dict] = []
        equity_rows = [{"timestamp": None, "balance": balance}]
        hold_signals = 0

        for index in range(0, len(featured) - self.expiration_candles):
            row = featured.iloc[index]
            probability_up = self._probability_up(pd.DataFrame([row[self.features].to_dict()]))
            signal, confidence = self._signal(probability_up)
            if signal == "HOLD":
                hold_signals += 1
                continue

            entry = float(row["close"])
            exit_row = featured.iloc[index + self.expiration_candles]
            exit_price = float(exit_row["close"])
            won = (signal == "BUY" and exit_price > entry) or (signal == "SELL" and exit_price < entry)
            pnl = self.stake * self.payout if won else -self.stake
            balance += pnl
            trades.append(
                {
                    "timestamp": row["timestamp"],
                    "signal": signal,
                    "confidence": confidence,
                    "probability_up": probability_up,
                    "entry": entry,
                    "exit": exit_price,
                    "won": won,
                    "stake": self.stake,
                    "pnl": pnl,
                    "balance": balance,
                }
            )
            equity_rows.append({"timestamp": exit_row["timestamp"], "balance": balance})

        trades_df = pd.DataFrame(trades, columns=ML_TRADE_COLUMNS)
        equity_df = pd.DataFrame(equity_rows)
        metrics = self._metrics(trades_df, equity_df, balance, hold_signals)
        self._save(trades_df, equity_df)
        return metrics, trades_df

    def _signal(self, probability_up: float) -> tuple[str, float]:
        if probability_up >= self.min_confidence:
            return "BUY", probability_up
        if probability_up <= 1 - self.min_confidence:
            return "SELL", 1 - probability_up
        return "HOLD", max(probability_up, 1 - probability_up)

    def _probability_up(self, X: pd.DataFrame) -> float:
        if hasattr(self.model, "predict_proba"):
            classes = list(self.model.classes_)
            probabilities = self.model.predict_proba(X)[0]
            if 1 in classes:
                return float(probabilities[classes.index(1)])
        return float(int(self.model.predict(X)[0]))

    def _metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        final_balance: float,
        hold_signals: int,
    ) -> MLBacktestMetrics:
        if trades.empty:
            return MLBacktestMetrics(
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
                average_profit_per_trade=0.0,
                total_hold_signals=hold_signals,
            )

        pnl = trades["pnl"]
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        total_trades = len(trades)
        return MLBacktestMetrics(
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            net_profit=float(pnl.sum()),
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=wins / total_trades,
            profit_factor=float(profit_factor(pnl)),
            max_drawdown=max_drawdown(equity_curve["balance"].astype(float)),
            max_consecutive_losses=max_consecutive_losses(pnl),
            average_profit_per_trade=float(pnl.mean()),
            total_hold_signals=hold_signals,
        )

    def _save(self, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(self.output_dir / "ml_backtest_results.csv", index=False)
        equity_curve.to_csv(self.output_dir / "ml_equity_curve.csv", index=False)
