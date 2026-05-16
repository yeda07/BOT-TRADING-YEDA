from types import SimpleNamespace

import pandas as pd

from app.brokers.paper_broker import PaperBroker
from app.execution.live_engine import LiveTradingEngine
from app.execution.order_manager import OrderManager
from app.execution.trade_logger import TradeLogger
from app.market.candle_storage import CandleStorage
from app.market.mock_realtime_feed import MockRealtimeDataFeed
from app.risk.risk_manager import RiskManager
from app.storage.trades_repository import TradesRepository


class AlwaysBuyPredictor:
    def predict_row(self, row):
        return {"signal": "BUY", "confidence": 0.9, "probability_up": 0.9}


def write_candles(path, rows: int = 90):
    prices = [100 + index * 0.1 for index in range(rows)]
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-10", periods=rows, freq="min", tz="UTC"),
            "open": prices,
            "high": [price + 0.2 for price in prices],
            "low": [price - 0.2 for price in prices],
            "close": [price + 0.05 for price in prices],
            "volume": [100] * rows,
        }
    ).to_csv(path, index=False)


def test_live_engine_runs_with_mock_realtime_feed_and_stores_candles(tmp_path):
    csv_path = tmp_path / "candles.csv"
    write_candles(csv_path)
    broker = PaperBroker(initial_balance=1000)
    broker.connect()
    settings = SimpleNamespace(
        ASSET="EURUSD-OTC",
        BOT_MODE="paper",
        MIN_CANDLES=60,
        LIVE_MAX_STEPS=65,
        LIVE_SLEEP_SECONDS=0,
    )
    storage = CandleStorage(str(tmp_path / "collected.csv"))
    repository = TradesRepository(str(tmp_path / "trades.db"))
    engine = LiveTradingEngine(
        data_feed=MockRealtimeDataFeed(str(csv_path), window_size=70, max_steps=65, cursor_path=str(tmp_path / "feed_cursor.json")),
        predictor=AlwaysBuyPredictor(),
        order_manager=OrderManager(
            broker,
            RiskManager(balance=1000, min_model_confidence=0.58, min_candles=60),
            payout=0.87,
            expiration_candles=1,
        ),
        trade_logger=TradeLogger(str(tmp_path / "live_trades.csv")),
        trades_repository=repository,
        settings=settings,
        candle_storage=storage,
    )

    engine.run()

    assert len(storage.read_all()) >= 60
    assert repository.get_summary()["total_trades"] >= 1
