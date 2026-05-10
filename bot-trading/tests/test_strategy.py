import pandas as pd

from app.strategies.rule_based import RuleBasedStrategy


def test_rule_strategy_buy_signal():
    row = pd.Series(
        {
            "sma_fast": 105,
            "sma_slow": 100,
            "rsi": 55,
            "macd": 1.2,
            "macd_signal": 0.8,
            "adx": 25,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "BUY"


def test_rule_strategy_holds_lateral_market():
    row = pd.Series(
        {
            "sma_fast": 105,
            "sma_slow": 100,
            "rsi": 55,
            "macd": 1.2,
            "macd_signal": 0.8,
            "adx": 10,
        }
    )
    signal = RuleBasedStrategy().generate_signal(row)
    assert signal.signal == "HOLD"

