from pathlib import Path

import pandas as pd


TRADE_LOG_COLUMNS = [
    "timestamp",
    "asset",
    "signal",
    "confidence",
    "amount",
    "entry_price",
    "exit_price",
    "result",
    "profit",
    "balance",
    "reason",
    "mode",
]


class TradeLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            pd.DataFrame(columns=TRADE_LOG_COLUMNS).to_csv(self.log_path, index=False)

    def log_trade(self, trade: dict) -> None:
        row = {column: trade.get(column) for column in TRADE_LOG_COLUMNS}
        frame = pd.DataFrame([row], columns=TRADE_LOG_COLUMNS)
        frame.to_csv(self.log_path, mode="a", header=False, index=False)

    def read_trades(self) -> pd.DataFrame:
        return pd.read_csv(self.log_path)
