from app.validation.overfitting_detector import OverfittingDetector


def test_overfitting_detector_detects_train_test_gap():
    result = OverfittingDetector().analyze(
        {"accuracy": 0.95, "roc_auc": 0.95},
        {"accuracy": 0.70, "roc_auc": 0.70},
        {"accuracy": 0.60, "roc_auc": 0.60, "total_trades": 20, "win_rate": 0.8},
        {"profitable_folds_ratio": 0.2, "std_win_rate": 0.2, "average_drawdown": 0.2},
    )

    assert result["is_overfitted"] is True
    assert result["risk_level"] == "HIGH"
