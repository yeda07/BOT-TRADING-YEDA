import pandas as pd

from app.validation.monte_carlo import MonteCarloSimulator


def test_monte_carlo_simulator_generates_percentiles():
    result = MonteCarloSimulator(pd.DataFrame({"pnl": [1, -1, 1, 1, -1]}), simulations=20).run()

    summary = result["summary"]
    assert "profit_p5" in summary
    assert "profit_p50" in summary
    assert "profit_p95" in summary
