from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BotMode = Literal["backtest", "paper", "demo", "real"]
BrokerName = Literal["paper", "demo_stub", "iqoption", "exnova"]
DataFeedSource = Literal["csv", "mock_realtime", "iqoption_demo", "exnova_demo"]


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
    MAX_ALLOWED_DRAWDOWN: float = Field(default=0.15, ge=0.0, le=1.0)
    MIN_PROFIT_FACTOR: float = Field(default=1.0, ge=0.0)
    MIN_TRADES_BEFORE_LIVE: int = Field(default=100, ge=0)

    # Backward-compatible aliases used by older configs.
    MAX_DAILY_LOSS_PCT: float = 0.05
    MIN_MODEL_CONFIDENCE: float = 0.58

    MIN_CANDLES: int = 200
    FEATURE_WINDOW_SIZE: int = Field(default=300, ge=60)
    MAX_VOLATILITY_MULTIPLIER: float = 3.0
    LATERAL_MARKET_ADX_THRESHOLD: float = 18.0

    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = Path("data/raw")
    LOG_DIR: Path = Path("data/logs")
    MODEL_DIR: Path = Path("models")
    DATABASE_URL: str = "sqlite:///data/logs/trades.db"
    DATA_FEED_SOURCE: DataFeedSource = "mock_realtime"
    CANDLES_CSV_PATH: str = "data/raw/candles.csv"
    COLLECTED_CANDLES_PATH: str = "data/raw/collected_candles.csv"
    LIVE_MAX_STEPS: int | None = Field(default=400, ge=1)
    LIVE_SLEEP_SECONDS: float = Field(default=1.0, ge=0.0)
    TRADE_LOG_PATH: str = "data/logs/live_trades.csv"
    TRADES_DB_PATH: str = "data/logs/trades.db"
    FEED_CURSOR_PATH: str = "data/logs/feed_cursor.json"
    RESET_FEED_CURSOR: bool = False
    RANDOM_FEED_START: bool = False
    FEED_START_INDEX: int | None = None
    ALLOW_REPLAY_SAME_WINDOW: bool = False
    KILL_SWITCH_PATH: str = "data/logs/kill_switch.json"
    EXECUTION_STATE_PATH: str = "data/logs/execution_state.json"
    ALERTS_LOG_PATH: str = "data/logs/alerts.log"
    DEMO_INITIAL_BALANCE: float = 3000.0
    VALIDATION_TRAIN_WINDOW: int = Field(default=2000, ge=100)
    VALIDATION_TEST_WINDOW: int = Field(default=500, ge=50)
    VALIDATION_STEP_SIZE: int = Field(default=250, ge=1)
    VALIDATION_N_SPLITS: int = Field(default=5, ge=2)
    VALIDATION_GAP: int = Field(default=1, ge=0)
    MIN_TRADES_FOR_THRESHOLD: int = Field(default=100, ge=1)
    MIN_PROFITABLE_FOLDS_RATIO: float = Field(default=0.60, ge=0.0, le=1.0)
    MAX_OVERFITTING_GAP: float = Field(default=0.10, ge=0.0, le=1.0)
    MIN_STABILITY_SCORE: float = Field(default=0.60, ge=0.0, le=1.0)
    MONTE_CARLO_SIMULATIONS: int = Field(default=1000, ge=1)
    STRESS_TEST_ENABLED: bool = True
    SESSION_STATE_PATH: str = "data/logs/current_session.json"
    RUNTIME_STATE_PATH: str = "data/logs/runtime_state.json"
    MODEL_REGISTRY_PATH: str = "models/model_registry.json"
    MODEL_VERSIONS_DIR: str = "models/versions"
    PRODUCTION_MODEL_PATH: str = "models/best_model.joblib"
    REPORTS_DIR: str = "data/reports"
    RETRAIN_MIN_NEW_CANDLES: int = Field(default=1000, ge=1)
    AUTO_PROMOTE_MODELS: bool = False
    DRIFT_MIN_TRADES: int = Field(default=100, ge=1)
    DRIFT_WIN_RATE_DROP: float = Field(default=0.05, ge=0.0, le=1.0)
    DRIFT_PROFIT_FACTOR_DROP: float = Field(default=0.20, ge=0.0)
    SUPERVISOR_HEARTBEAT_SECONDS: int = Field(default=30, ge=1)
    MAX_RUNTIME_ERRORS: int = Field(default=5, ge=1)

    IQOPTION_EMAIL: str | None = None
    IQOPTION_PASSWORD: str | None = None
    EXNOVA_EMAIL: str | None = None
    EXNOVA_PASSWORD: str | None = None

    @model_validator(mode="after")
    def validate_trading_mode(self) -> "Settings":
        if self.BOT_MODE == "real" and not self.ENABLE_REAL_TRADING:
            raise RuntimeError("Real trading is disabled by default.")
        return self

    @field_validator("FEED_START_INDEX", mode="before")
    @classmethod
    def empty_feed_start_index_is_none(cls, value):
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
