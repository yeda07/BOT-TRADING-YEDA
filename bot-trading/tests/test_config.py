import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_accepts_allowed_modes():
    for mode in ["backtest", "paper", "demo"]:
        assert Settings(BOT_MODE=mode).BOT_MODE == mode


def test_settings_blocks_real_mode_by_default():
    with pytest.raises(RuntimeError, match="Real trading is disabled"):
        Settings(BOT_MODE="real", ENABLE_REAL_TRADING=False)


def test_settings_allows_real_mode_when_explicitly_enabled():
    settings = Settings(BOT_MODE="real", ENABLE_REAL_TRADING=True)
    assert settings.BOT_MODE == "real"


def test_settings_rejects_risk_per_trade_above_two_percent():
    with pytest.raises(ValidationError):
        Settings(RISK_PER_TRADE=0.021)


def test_settings_rejects_daily_loss_above_ten_percent():
    with pytest.raises(ValidationError):
        Settings(MAX_DAILY_LOSS=0.11)
