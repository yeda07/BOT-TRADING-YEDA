import pytest

from app.utils.metrics import breakeven_win_rate


def test_breakeven_win_rate_for_binary_options():
    assert breakeven_win_rate(0.87) == pytest.approx(0.534759, rel=1e-5)


def test_breakeven_win_rate_rejects_invalid_payout():
    with pytest.raises(ValueError):
        breakeven_win_rate(0)
