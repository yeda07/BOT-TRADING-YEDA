import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from app.utils.metrics import breakeven_win_rate, max_drawdown, profit_factor


def probability_up(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(X)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
    return model.predict(X).astype(float)


def trading_metrics_from_probabilities(probabilities, y, payout: float, threshold: float = 0.58) -> dict:
    y_array = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    trade_mask = (probabilities >= threshold) | (probabilities <= 1 - threshold)
    if not trade_mask.any():
        return _empty_metrics(payout)
    predictions = np.where(probabilities[trade_mask] >= threshold, 1, 0)
    actual = y_array[trade_mask]
    wins = predictions == actual
    pnl = np.where(wins, payout, -1.0)
    pnl_series = pd.Series(pnl)
    equity = 100.0 + pnl_series.cumsum()
    total = len(pnl)
    win_rate = float(wins.mean()) if total else 0.0
    breakeven = breakeven_win_rate(payout)
    return {
        "total_trades": int(total),
        "win_rate": win_rate,
        "breakeven_win_rate": breakeven,
        "edge": win_rate - breakeven,
        "net_profit": float(pnl.sum()),
        "profit_factor": float(profit_factor(pnl_series)),
        "max_drawdown": float(max_drawdown(equity)),
        "average_profit_per_trade": float(pnl.mean()) if total else 0.0,
    }


def classification_metrics(model, X: pd.DataFrame, y) -> dict:
    predictions = model.predict(X)
    try:
        roc_auc = float(roc_auc_score(y, probability_up(model, X))) if len(set(y)) > 1 else 0.0
    except ValueError:
        roc_auc = 0.0
    return {
        "roc_auc": roc_auc if not math.isnan(roc_auc) else 0.0,
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
    }


def composite_score(metrics: dict) -> float:
    roc_auc = max(0.0, min(1.0, float(metrics.get("roc_auc", 0.0))))
    pf = min(float(metrics.get("profit_factor", 0.0)), 3.0) / 3.0
    edge = max(-0.2, min(0.2, float(metrics.get("edge", 0.0))))
    edge_norm = (edge + 0.2) / 0.4
    f1 = max(0.0, min(1.0, float(metrics.get("f1", 0.0))))
    dd = max(0.0, min(1.0, float(metrics.get("max_drawdown", 0.0))))
    return 0.35 * roc_auc + 0.25 * pf + 0.20 * edge_norm + 0.10 * f1 - 0.10 * dd


def _empty_metrics(payout: float) -> dict:
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "breakeven_win_rate": breakeven_win_rate(payout),
        "edge": -breakeven_win_rate(payout),
        "net_profit": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "average_profit_per_trade": 0.0,
    }
