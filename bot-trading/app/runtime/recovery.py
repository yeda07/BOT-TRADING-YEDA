class RecoveryManager:
    def __init__(self, runtime_state, trades_repository, settings):
        self.runtime_state = runtime_state
        self.trades_repository = trades_repository
        self.settings = settings

    def recover(self) -> dict:
        trades = self.trades_repository.get_all_trades()
        last_trade = trades[-1] if trades else None
        state = self.runtime_state.load()
        if last_trade:
            state.update(
                {
                    "last_candle_timestamp": last_trade.get("timestamp"),
                    "last_signal": last_trade.get("signal"),
                    "last_trade_result": last_trade.get("result"),
                    "current_balance": last_trade.get("balance") or state.get("current_balance", 0.0),
                    "total_trades": len([trade for trade in trades if trade.get("result") in {"WON", "LOST"}]),
                }
            )
        self.runtime_state.save(state)
        return {"state": state, "last_trade": last_trade, "skip_timestamp": state.get("last_candle_timestamp")}
