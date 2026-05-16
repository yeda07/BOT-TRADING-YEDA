import time

from app.market.data_quality import validate_candles_df
from app.market.features import build_features


class LiveTradingEngine:
    def __init__(
        self,
        data_feed,
        predictor,
        order_manager,
        trade_logger,
        trades_repository,
        settings,
        candle_storage=None,
        session_id: str = "manual",
    ):
        self.data_feed = data_feed
        self.predictor = predictor
        self.order_manager = order_manager
        self.trade_logger = trade_logger
        self.trades_repository = trades_repository
        self.settings = settings
        self.candle_storage = candle_storage
        self.session_id = session_id or "manual"

    def run(self) -> dict:
        print("===== LIVE PAPER TRADING =====")
        print(f"Asset: {self.settings.ASSET}")
        print(f"Mode: {self.settings.BOT_MODE}")
        print(f"Session: {self.session_id}")
        print(f"Balance: {self.order_manager.broker.get_balance():.2f}")

        steps = 0
        max_steps = self.settings.LIVE_MAX_STEPS
        required_window = self._required_window()
        self.data_feed.connect()
        try:
            while self.data_feed.has_next() and (max_steps is None or steps < max_steps):
                latest_candle = self.data_feed.get_next_candle()
                if self.candle_storage is not None:
                    self.candle_storage.append_candle(latest_candle)
                window = self.data_feed.get_latest_candles()
                steps += 1

                if len(window) < required_window:
                    print(f"Warm-up: current candles {len(window)} / required {required_window}")
                    continue

                valid, errors = validate_candles_df(window) if not window.empty else (False, ["No candles available."])
                if not valid:
                    print(f"WARNING: Skipping corrupt candle/window: {'; '.join(errors)}")
                    continue

                trade = self._process_window(window, latest_candle)
                if trade is not None:
                    self.trade_logger.log_trade(trade)
                    self.trades_repository.insert_trade(trade)

                if self.settings.LIVE_SLEEP_SECONDS > 0 and self.data_feed.has_next():
                    time.sleep(self.settings.LIVE_SLEEP_SECONDS)
        finally:
            self.data_feed.disconnect()
        return self._session_feed_metadata()

    def _process_window(self, window, latest_candle) -> dict | None:
        required_window = self._required_window()
        if len(window) < required_window:
            result = {
                "status": "SKIPPED",
                "signal": "HOLD",
                "confidence": 0.0,
                "amount": 0.0,
                "reason": f"Not enough candles for live decision. Need {required_window}.",
                "timestamp": latest_candle.get("timestamp"),
                "entry_price": latest_candle.get("close"),
            }
            self._print_skipped(result)
            return None

        featured = build_features(window).dropna()
        if featured.empty:
            result = {
                "status": "SKIPPED",
                "signal": "HOLD",
                "confidence": 0.0,
                "amount": 0.0,
                "reason": "No valid feature row for live decision.",
                "timestamp": latest_candle.get("timestamp"),
                "entry_price": latest_candle.get("close"),
            }
            self._print_skipped(result)
            return None

        feature_row = featured.iloc[-1]
        prediction = self.predictor.predict_row(feature_row)
        execution = self.order_manager.execute_signal(
            signal=str(prediction["signal"]),
            confidence=float(prediction["confidence"]),
            asset=self.settings.ASSET,
            candle=feature_row,
        )

        if execution["status"] != "EXECUTED":
            self._print_skipped(execution)
            return self._trade_record(
                execution,
                exit_price=None,
                result=execution["status"],
                profit=0.0,
                balance=self.order_manager.broker.get_balance(),
            )

        exit_candle = self._consume_expiration_candle()
        if exit_candle is None:
            self._print_skipped({**execution, "reason": "No future candle available to resolve order."})
            return self._trade_record(
                execution,
                exit_price=None,
                result="PENDING",
                profit=0.0,
                balance=self.order_manager.broker.get_balance(),
            )

        resolution = self.order_manager.broker.resolve_order(
            execution["order_id"],
            entry_price=execution["entry_price"],
            exit_price=float(exit_candle["close"]),
            payout=self.order_manager.payout,
        )
        self.order_manager.risk_manager.register_result(float(resolution["profit"]))
        trade = self._trade_record(
            execution,
            exit_price=resolution["exit_price"],
            result=resolution["status"],
            profit=resolution["profit"],
            balance=resolution["balance"],
        )
        self._print_executed(execution, resolution)
        return trade

    def _consume_expiration_candle(self):
        exit_candle = None
        for _ in range(self.order_manager.expiration_candles):
            if not self.data_feed.has_next():
                return None
            exit_candle = self.data_feed.get_next_candle()
            if self.candle_storage is not None:
                self.candle_storage.append_candle(exit_candle)
        return exit_candle

    def _trade_record(self, execution: dict, exit_price, result: str, profit: float, balance: float) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": execution.get("timestamp"),
            "asset": self.settings.ASSET,
            "signal": execution.get("signal"),
            "confidence": execution.get("confidence"),
            "amount": execution.get("amount"),
            "entry_price": execution.get("entry_price"),
            "exit_price": exit_price,
            "result": result,
            "profit": profit,
            "balance": balance,
            "reason": execution.get("reason"),
            "mode": self.settings.BOT_MODE,
        }

    def _print_skipped(self, execution: dict) -> None:
        print(f"Latest candle: {execution.get('timestamp')}")
        print(f"Signal: {execution.get('signal')}")
        print(f"Confidence: {float(execution.get('confidence', 0.0)):.2f}")
        print(f"Status: {execution.get('status')}")
        print(f"Reason: {execution.get('reason')}")

    def _print_executed(self, execution: dict, resolution: dict) -> None:
        print(f"Latest candle: {execution.get('timestamp')}")
        print(f"Signal: {execution.get('signal')}")
        print(f"Confidence: {float(execution.get('confidence', 0.0)):.2f}")
        print(f"Status: {execution.get('status')}")
        print(f"Result: {resolution.get('status')}")
        print(f"Profit: {float(resolution.get('profit', 0.0)):.2f}")
        print(f"New Balance: {float(resolution.get('balance', 0.0)):.2f}")

    def _required_window(self) -> int:
        min_candles = int(getattr(self.settings, "MIN_CANDLES", 200))
        feature_window = int(getattr(self.settings, "FEATURE_WINDOW_SIZE", min_candles))
        return max(min_candles, feature_window)

    def _session_feed_metadata(self) -> dict:
        data = getattr(self.data_feed, "_data", None)
        start_index = getattr(self.data_feed, "start_index", None)
        end_index = getattr(self.data_feed, "end_index", None)
        candle_count = 0
        start_timestamp = None
        end_timestamp = None
        if data is not None and start_index is not None and end_index is not None and end_index >= start_index:
            candle_count = int(end_index - start_index + 1)
            start_timestamp = str(data.iloc[start_index].get("timestamp"))
            end_timestamp = str(data.iloc[end_index].get("timestamp"))
        return {
            "data_start_timestamp": start_timestamp,
            "data_end_timestamp": end_timestamp,
            "feed_start_index": start_index,
            "feed_end_index": end_index,
            "candle_count_processed": candle_count,
        }
