from types import SimpleNamespace

from app.mlops.model_drift_detector import ModelDriftDetector
from app.storage.trades_repository import TradesRepository


def test_model_drift_detector_detects_win_rate_drop(tmp_path):
    repo = TradesRepository(str(tmp_path / "trades.db"))
    for index in range(10):
        repo.insert_trade(
            {
                "timestamp": str(index),
                "asset": "EURUSD-OTC",
                "signal": "BUY",
                "confidence": 0.4,
                "amount": 10,
                "entry_price": 100,
                "exit_price": 99,
                "result": "LOST",
                "profit": -10,
                "balance": 1000 - index * 10,
                "reason": "test",
                "mode": "paper",
            }
        )
    settings = SimpleNamespace(DRIFT_MIN_TRADES=5, PAYOUT=0.87, DRIFT_WIN_RATE_DROP=0.05, MIN_PROFIT_FACTOR=1.0, DRIFT_PROFIT_FACTOR_DROP=0.2, MIN_CONFIDENCE=0.58)

    result = ModelDriftDetector(repo, settings).analyze()

    assert result["drift_detected"]
    assert result["risk_level"] in {"MEDIUM", "HIGH"}
