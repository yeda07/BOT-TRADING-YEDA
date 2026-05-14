import pandas as pd


class OrderManager:
    def __init__(self, broker, risk_manager, payout: float, expiration_candles: int):
        if expiration_candles < 1:
            raise ValueError("expiration_candles must be at least 1.")
        self.broker = broker
        self.risk_manager = risk_manager
        self.payout = payout
        self.expiration_candles = expiration_candles
        self._traded_timestamps: set[str] = set()

    def execute_signal(
        self,
        signal: str,
        confidence: float,
        asset: str,
        candle: pd.Series,
    ) -> dict:
        timestamp = str(candle.get("timestamp", ""))
        base = {
            "status": "SKIPPED",
            "signal": signal,
            "confidence": float(confidence),
            "amount": 0.0,
            "reason": "",
            "timestamp": candle.get("timestamp"),
            "entry_price": float(candle.get("close", 0.0)),
            "order_id": None,
        }

        if signal == "HOLD":
            return {**base, "status": "HOLD", "reason": "Signal is HOLD."}
        if confidence < self.risk_manager.min_model_confidence:
            return {**base, "status": "LOW_CONFIDENCE", "reason": "Confidence below minimum threshold."}
        if timestamp in self._traded_timestamps:
            return {**base, "status": "DUPLICATED_CANDLE", "reason": "A trade was already opened for this candle."}
        if not self.risk_manager.can_trade():
            return {**base, "status": "BLOCKED_BY_RISK", "reason": "RiskManager does not allow trading."}

        amount = self.risk_manager.get_position_size()
        if amount <= 0:
            return {**base, "status": "BLOCKED_BY_RISK", "reason": "Position size is zero."}

        result = self.broker.place_order(asset, amount, signal, self.expiration_candles)
        if result.status != "accepted":
            return {**base, "status": "BLOCKED_BY_RISK", "amount": amount, "reason": result.message}

        self._traded_timestamps.add(timestamp)
        return {
            **base,
            "status": "EXECUTED",
            "amount": amount,
            "reason": "Order sent to paper broker and pending resolution.",
            "order_id": result.order_id,
        }
