class DemoExecutor:
    def __init__(self, broker, execution_guard, risk_manager, trade_logger, trades_repository, settings):
        self.broker = broker
        self.execution_guard = execution_guard
        self.risk_manager = risk_manager
        self.trade_logger = trade_logger
        self.trades_repository = trades_repository
        self.settings = settings

    def execute(self, signal: str, confidence: float, asset: str, candle) -> dict:
        timestamp = candle.get("timestamp")
        entry_price = float(candle.get("close", 0.0))
        allowed, reason = self.execution_guard.validate_before_order(
            signal, confidence, self.broker.get_balance(), asset, timestamp
        )
        if not allowed:
            return {"status": "BLOCKED", "reason": reason, "signal": signal, "timestamp": timestamp}
        if self.broker.get_account_type() != "demo":
            raise RuntimeError("Demo execution blocked because broker account is not demo.")

        amount = self.risk_manager.get_position_size()
        order = self.broker.place_demo_order(asset, amount, signal, self.settings.EXPIRATION_CANDLES * self.settings.TIMEFRAME_SECONDS)
        if order.get("status") == "ERROR":
            return {"status": "ERROR", "reason": order.get("reason", "Broker demo order failed.")}

        pending_trade = self._trade(order, timestamp, signal, confidence, entry_price, "PENDING", 0.0)
        self.trades_repository.insert_trade(pending_trade)
        self.execution_guard.mark_traded(asset, timestamp)

        status = self.broker.get_order_status(order["order_id"])
        result = status.get("result", status.get("status", "PENDING"))
        profit = float(status.get("profit", 0.0))
        self.risk_manager.register_result(profit)
        resolved = self._trade(order, timestamp, signal, confidence, entry_price, result, profit)
        self.trade_logger.log_trade(resolved)
        self.trades_repository.update_trade_result(order["order_id"], result, profit, self.broker.get_balance())
        return {"status": "RESOLVED", "order_id": order["order_id"], "result": result, "profit": profit}

    def _trade(self, order: dict, timestamp, signal: str, confidence: float, entry_price: float, result: str, profit: float) -> dict:
        return {
            "timestamp": timestamp,
            "asset": order["asset"],
            "signal": signal,
            "confidence": confidence,
            "amount": order["amount"],
            "entry_price": entry_price,
            "exit_price": None,
            "result": result,
            "profit": profit,
            "balance": self.broker.get_balance(),
            "reason": "Demo execution",
            "mode": "demo",
            "order_id": order["order_id"],
        }
