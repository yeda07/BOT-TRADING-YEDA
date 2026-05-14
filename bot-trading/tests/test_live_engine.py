from types import SimpleNamespace

import pandas as pd

from app.brokers.paper_broker import PaperBroker
from app.execution.live_engine import LiveTradingEngine
from app.execution.order_manager import OrderManager
from app.execution.trade_logger import TradeLogger
from app.market.data_feed import CSVDataFeed
from app.risk.risk_manager import RiskManager
from app.storage.trades_repository import TradesRepository


class AlwaysBuyPredictor:
    def predict_row(self, row):
        return {"signal": "BUY", "confidence": 0.9, "probability_up": 0.9}


def candles(rows: int = 90) -> pd.DataFrame:
    prices = [100 + index * 0.1 for index in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-10", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [price + 0.2 for price in prices],
            "low": [price - 0.2 for price in prices],
            "close": [price + 0.05 for price in prices],
            "volume": [100] * rows,
        }
    )


def test_live_trading_engine_runs_with_small_step_limit(tmp_path):
    csv_path = tmp_path / "candles.csv"
    candles().to_csv(csv_path, index=False)
    broker = PaperBroker(initial_balance=1000)
    broker.connect()
    risk_manager = RiskManager(balance=1000, min_model_confidence=0.58, min_candles=60)
    settings = SimpleNamespace(
        ASSET="EURUSD-OTC",
        BOT_MODE="paper",
        MIN_CANDLES=60,
        LIVE_MAX_STEPS=65,
        LIVE_SLEEP_SECONDS=0,
    )
    logger = TradeLogger(str(tmp_path / "live_trades.csv"))
    repository = TradesRepository(str(tmp_path / "trades.db"))
    engine = LiveTradingEngine(
        data_feed=CSVDataFeed(str(csv_path), window_size=70),
        predictor=AlwaysBuyPredictor(),
        order_manager=OrderManager(broker, risk_manager, payout=0.87, expiration_candles=1),
        trade_logger=logger,
        trades_repository=repository,
        settings=settings,
    )

    engine.run()

    summary = repository.get_summary()
    assert summary["total_trades"] >= 1
    assert summary["wins"] >= 1
    assert logger.read_trades()["result"].isin(["WON"]).any()
