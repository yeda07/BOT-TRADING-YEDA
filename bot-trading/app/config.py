from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BotMode = Literal["backtest", "paper", "demo", "real"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "BOT-YEDA Trading Bot"
    BOT_MODE: BotMode = "backtest"
    ENABLE_REAL_TRADING: bool = False

    INITIAL_BALANCE: float = 1000.0
    RISK_PER_TRADE: float = Field(default=0.01, ge=0.0, le=0.05)
    MAX_CONSECUTIVE_LOSSES: int = 3
    MAX_DAILY_LOSS_PCT: float = 0.05
    MIN_MODEL_CONFIDENCE: float = 0.60
    MIN_CANDLES: int = 50
    MAX_VOLATILITY_MULTIPLIER: float = 3.0
    LATERAL_MARKET_ADX_THRESHOLD: float = 18.0

    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = Path("data/raw")
    LOG_DIR: Path = Path("data/logs")
    MODEL_DIR: Path = Path("models")
    DATABASE_URL: str = "sqlite:///data/logs/trades.db"

    IQOPTION_EMAIL: str | None = None
    IQOPTION_PASSWORD: str | None = None
    EXNOVA_EMAIL: str | None = None
    EXNOVA_PASSWORD: str | None = None

    def validate_trading_mode(self) -> None:
        if self.BOT_MODE == "real" and not self.ENABLE_REAL_TRADING:
            raise RuntimeError("Real trading is disabled by default.")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_trading_mode()
    return settings

