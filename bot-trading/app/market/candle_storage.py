from pathlib import Path

import pandas as pd

from app.market.candle_schema import STANDARD_CANDLE_COLUMNS, normalize_candles
from app.market.data_quality import clean_candles_df


class CandleStorage:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            pd.DataFrame(columns=STANDARD_CANDLE_COLUMNS).to_csv(self.csv_path, index=False)

    def append_candle(self, candle: dict | pd.Series) -> None:
        self.append_candles(pd.DataFrame([dict(candle)]))

    def append_candles(self, candles: pd.DataFrame) -> None:
        if candles.empty:
            return
        asset = str(candles["asset"].dropna().iloc[0]) if "asset" in candles and not candles["asset"].dropna().empty else "UNKNOWN"
        timeframe = int(candles["timeframe_seconds"].dropna().iloc[0]) if "timeframe_seconds" in candles and not candles["timeframe_seconds"].dropna().empty else 60
        source = str(candles["source"].dropna().iloc[0]) if "source" in candles and not candles["source"].dropna().empty else "unknown"
        normalized = normalize_candles(candles, asset=asset, timeframe_seconds=timeframe, source=source)
        existing = self.read_all()
        combined = normalized if existing.empty else pd.concat([existing, normalized], ignore_index=True)
        clean_candles_df(combined).to_csv(self.csv_path, index=False)

    def read_all(self) -> pd.DataFrame:
        data = pd.read_csv(self.csv_path)
        if data.empty:
            return pd.DataFrame(columns=STANDARD_CANDLE_COLUMNS)
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
        return data[STANDARD_CANDLE_COLUMNS]

    def remove_duplicates(self) -> None:
        clean_candles_df(self.read_all()).to_csv(self.csv_path, index=False)
