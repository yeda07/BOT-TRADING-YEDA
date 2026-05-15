class OrderReconciliation:
    def __init__(self, broker, trades_repository):
        self.broker = broker
        self.trades_repository = trades_repository

    def reconcile_pending_orders(self) -> dict:
        checked = updated = errors = 0
        for trade in self.trades_repository.get_pending_trades():
            checked += 1
            order_id = trade.get("order_id")
            try:
                status = self.broker.get_order_status(order_id)
                result = status.get("result", status.get("status"))
                if result and result != "PENDING":
                    self.trades_repository.update_trade_result(
                        order_id,
                        result,
                        float(status.get("profit", 0.0)),
                        float(status.get("balance", trade.get("balance") or 0.0)),
                    )
                    updated += 1
            except Exception:
                errors += 1
        return {"checked": checked, "updated": updated, "errors": errors}
