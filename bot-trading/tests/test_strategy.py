import pandas as pd

from app.strategies.rule_based import RuleBasedStrategy


def test_rule_strategy_buy_signal():
    row = pd.Series(
        {
            "ema_9": 105,
            "ema_21": 100,
            "rsi": 55,
            "close": 106,
            "macd_diff": 0.4,
            "atr": 0.3,
            "adx": 25,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "BUY"
    assert signal.to_dict()["signal"] == "BUY"


def test_rule_strategy_sell_signal():
    row = pd.Series(
        {
            "ema_9": 95,
            "ema_21": 100,
            "rsi": 45,
            "close": 94,
            "macd_diff": -0.4,
            "atr": 0.3,
            "adx": 25,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "SELL"


def test_rule_strategy_holds_lateral_market():
    row = pd.Series(
        {
            "ema_9": 105,
            "ema_21": 100,
            "rsi": 55,
            "close": 106,
            "macd_diff": 0.4,
            "atr": 0.3,
            "adx": 10,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "HOLD"


def test_rule_strategy_holds_extreme_rsi():
    row = pd.Series(
        {
            "ema_9": 105,
            "ema_21": 100,
            "rsi": 75,
            "close": 106,
            "macd_diff": 0.4,
            "atr": 0.3,
            "adx": 25,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "HOLD"
    assert "RSI" in signal.reason


def test_rule_strategy_holds_low_volatility():
    row = pd.Series(
        {
            "ema_9": 105,
            "ema_21": 100,
            "rsi": 55,
            "close": 106,
            "macd_diff": 0.4,
            "atr": 0.0,
            "adx": 25,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "HOLD"
    assert "volatility" in signal.reason


def test_rule_strategy_holds_mixed_signals():
    row = pd.Series(
        {
            "ema_9": 105,
            "ema_21": 100,
            "rsi": 55,
            "close": 99,
            "macd_diff": -0.4,
            "atr": 0.3,
            "adx": 25,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "HOLD"
    assert "Mixed" in signal.reason


def test_rule_strategy_holds_insufficient_data():
    row = pd.Series({"ema_9": 105, "ema_21": 100})
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "HOLD"
    assert "Insufficient" in signal.reason
