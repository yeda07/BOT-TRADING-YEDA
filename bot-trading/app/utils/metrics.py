import math

import pandas as pd


def breakeven_win_rate(payout: float) -> float:
    if payout <= 0:
        raise ValueError("payout must be greater than zero.")
    return 1 / (1 + payout)


def profit_factor(profits: pd.Series) -> float:
    gross_profit = float(profits[profits > 0].sum())
    gross_loss = abs(float(profits[profits < 0].sum()))
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    drawdowns = (peaks - equity) / peaks.replace(0, pd.NA)
    return float(drawdowns.fillna(0.0).max())


def max_consecutive_losses(profits: pd.Series) -> int:
    current = 0
    longest = 0
    for pnl in profits:
        if pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
