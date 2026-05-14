from app.brokers.exnova_broker import ExnovaBroker
from app.brokers.iqoption_broker import IQOptionBroker
from app.market.data_feed import CSVDataFeed
from app.market.mock_realtime_feed import MockRealtimeDataFeed


class DataFeedManager:
    def __init__(self, settings):
        self.settings = settings

    def create_feed(self):
        source = self.settings.DATA_FEED_SOURCE
        if source == "csv":
            return CSVDataFeed(
                csv_path=self.settings.CANDLES_CSV_PATH,
                window_size=max(self.settings.MIN_CANDLES, 300),
                asset=self.settings.ASSET,
                timeframe_seconds=self.settings.TIMEFRAME_SECONDS,
            )
        if source == "mock_realtime":
            return MockRealtimeDataFeed(
                csv_path=self.settings.CANDLES_CSV_PATH,
                window_size=max(self.settings.MIN_CANDLES, 300),
                asset=self.settings.ASSET,
                timeframe_seconds=self.settings.TIMEFRAME_SECONDS,
                sleep_seconds=0.0,
                max_steps=self.settings.LIVE_MAX_STEPS,
            )
        if source == "iqoption_demo":
            raise NotImplementedError(IQOptionBroker.adapter_not_implemented_message())
        if source == "exnova_demo":
            raise NotImplementedError(ExnovaBroker.adapter_not_implemented_message())
        raise ValueError(f"Unsupported DATA_FEED_SOURCE: {source}")
