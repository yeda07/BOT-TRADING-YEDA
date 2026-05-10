from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BotMode = Literal["backtest", "paper", "demo", "real"]
BrokerName = Literal["paper", "iqoption", "exnova"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "BOT-YEDA Trading Bot"
    BOT_MODE: BotMode = "backtest"
    BROKER: BrokerName = "paper"
    ENABLE_REAL_TRADING: bool = False

    ASSET: str = "EURUSD-OTC"
    TIMEFRAME_SECONDS: int = Field(default=60, ge=1)
    EXPIRATION_CANDLES: int = Field(default=1, ge=1)

    INITIAL_BALANCE: float = 3000.0
    PAYOUT: float = Field(default=0.87, gt=0.0)
    RISK_PER_TRADE: float = Field(default=0.01, ge=0.0, le=0.02)
    MAX_DAILY_LOSS: float = Field(default=0.05, ge=0.0, le=0.10)
    MAX_CONSECUTIVE_LOSSES: int = Field(default=3, ge=0)
    MIN_CONFIDENCE: float = Field(default=0.58, ge=0.0, le=1.0)

    # Backward-compatible aliases used by older configs.
    MAX_DAILY_LOSS_PCT: float = 0.05
    MIN_MODEL_CONFIDENCE: float = 0.58

    MIN_CANDLES: int = 200
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

    @model_validator(mode="after")
    def validate_trading_mode(self) -> "Settings":
        if self.BOT_MODE == "real" and not self.ENABLE_REAL_TRADING:
            raise RuntimeError("Real trading is disabled by default.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
