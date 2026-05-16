from pathlib import Path

import pandas as pd


TRADE_LOG_COLUMNS = [
    "session_id",
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
        else:
            self._migrate_columns()

    def log_trade(self, trade: dict) -> None:
        row = {column: trade.get(column) for column in TRADE_LOG_COLUMNS}
        row["session_id"] = row.get("session_id") or "legacy"
        frame = pd.DataFrame([row], columns=TRADE_LOG_COLUMNS)
        frame.to_csv(self.log_path, mode="a", header=False, index=False)

    def read_trades(self) -> pd.DataFrame:
        return pd.read_csv(self.log_path)

    def _migrate_columns(self) -> None:
        data = pd.read_csv(self.log_path)
        changed = False
        for column in TRADE_LOG_COLUMNS:
            if column not in data.columns:
                data[column] = "legacy" if column == "session_id" else None
                changed = True
        if changed or list(data.columns) != TRADE_LOG_COLUMNS:
            data = data[TRADE_LOG_COLUMNS]
            data.to_csv(self.log_path, index=False)
