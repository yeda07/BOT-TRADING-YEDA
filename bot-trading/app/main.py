from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.execution.backtester import Backtester
from app.execution.comparison import compare_rule_vs_ml
from app.market.candles import load_candles_csv, validate_candles
from app.ml.train import train_model
from app.risk.risk_manager import RiskManager
from app.storage.database import save_trades
from app.strategies.rule_based import RuleBasedStrategy
from app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(log_dir=settings.LOG_DIR)
app = FastAPI(title=settings.PROJECT_NAME, version="0.1.0")


class BacktestRequest(BaseModel):
    csv_path: str
    initial_balance: float | None = None
    save_to_db: bool = True


class TrainRequest(BaseModel):
    csv_path: str
    output_path: str = "models/best_model.joblib"


class CompareRequest(BaseModel):
    csv_path: str
    model_path: str = "models/model.joblib"
    initial_balance: float | None = None
    save_to_db: bool = True


def make_risk_manager() -> RiskManager:
    return RiskManager(
        risk_per_trade=settings.RISK_PER_TRADE,
        max_consecutive_losses=settings.MAX_CONSECUTIVE_LOSSES,
        max_daily_loss_pct=settings.MAX_DAILY_LOSS,
        min_model_confidence=settings.MIN_CONFIDENCE,
        min_candles=settings.MIN_CANDLES,
        max_volatility_multiplier=settings.MAX_VOLATILITY_MULTIPLIER,
        lateral_market_adx_threshold=settings.LATERAL_MARKET_ADX_THRESHOLD,
    )


@app.get("/health")
def health() -> dict[str, str]:
    settings.validate_trading_mode()
    return {"status": "ok", "mode": settings.BOT_MODE}


@app.post("/backtest")
def backtest(request: BacktestRequest) -> dict:
    try:
        candles = load_candles_csv(request.csv_path)
        validate_candles(candles, min_rows=settings.MIN_CANDLES)
        tester = Backtester(
            strategy=RuleBasedStrategy(),
            risk_manager=make_risk_manager(),
            initial_balance=request.initial_balance or settings.INITIAL_BALANCE,
            payout=settings.PAYOUT,
            expiration_candles=settings.EXPIRATION_CANDLES,
            logger=logger,
            strategy_name="rule_based",
        )
        metrics, trades = tester.run(candles)
        saved = save_trades(trades) if request.save_to_db else 0
        logger.info("Backtest completed for %s with %s trades", request.csv_path, metrics.trades)
        return {"metrics": metrics.__dict__, "saved_trades": saved}
    except Exception as exc:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train")
def train(request: TrainRequest) -> dict[str, float]:
    try:
        candles = load_candles_csv(request.csv_path)
        validate_candles(candles, min_rows=settings.MIN_CANDLES)
        metrics = train_model(
            candles,
            output_path=request.output_path,
            expiration_candles=settings.EXPIRATION_CANDLES,
            payout=settings.PAYOUT,
        )
        logger.info("ML model trained and stored at %s", request.output_path)
        return metrics
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/compare-strategies")
def compare_strategies(request: CompareRequest) -> dict:
    try:
        candles = load_candles_csv(request.csv_path)
        validate_candles(candles, min_rows=settings.MIN_CANDLES)
        comparison, trades = compare_rule_vs_ml(
            candles=candles,
            model_path=request.model_path,
            risk_manager_factory=make_risk_manager,
            initial_balance=request.initial_balance or settings.INITIAL_BALANCE,
            payout=settings.PAYOUT,
            expiration_candles=settings.EXPIRATION_CANDLES,
            logger=logger,
        )
        saved = save_trades(trades) if request.save_to_db else 0
        logger.info(
            "Strategy comparison completed for %s with winner=%s",
            request.csv_path,
            comparison.winner,
        )
        return {"comparison": comparison.to_dict(), "saved_trades": saved}
    except Exception as exc:
        logger.exception("Strategy comparison failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run_backtest_cli(csv_path: str, output_csv: str | None = None) -> None:
    candles = load_candles_csv(csv_path)
    validate_candles(candles, min_rows=settings.MIN_CANDLES)
    tester = Backtester(
        RuleBasedStrategy(),
        make_risk_manager(),
        settings.INITIAL_BALANCE,
        payout=settings.PAYOUT,
        expiration_candles=settings.EXPIRATION_CANDLES,
        logger=logger,
        strategy_name="rule_based",
    )
    metrics, trades = tester.run(candles)
    print(pd.Series(metrics.__dict__).to_string())
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output_csv, index=False)


def run_compare_cli(csv_path: str, model_path: str) -> None:
    candles = load_candles_csv(csv_path)
    validate_candles(candles, min_rows=settings.MIN_CANDLES)
    comparison, _ = compare_rule_vs_ml(
        candles=candles,
        model_path=model_path,
        risk_manager_factory=make_risk_manager,
        initial_balance=settings.INITIAL_BALANCE,
        payout=settings.PAYOUT,
        expiration_candles=settings.EXPIRATION_CANDLES,
        logger=logger,
    )
    print(pd.Series(comparison.to_dict()).to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BOT-YEDA trading bot")
    parser.add_argument("command", choices=["backtest", "train", "compare"])
    parser.add_argument("--csv", required=True, help="Path to candles CSV")
    parser.add_argument("--output", default=None, help="Output CSV for trades or model path")
    parser.add_argument("--model", default="models/model.joblib", help="Model path for strategy comparison")
    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest_cli(args.csv, args.output)
    elif args.command == "train":
        output = args.output or "models/best_model.joblib"
        candles = load_candles_csv(args.csv)
        validate_candles(candles, min_rows=settings.MIN_CANDLES)
        metrics = train_model(candles, output, expiration_candles=settings.EXPIRATION_CANDLES, payout=settings.PAYOUT)
        print(pd.Series(metrics).to_string())
    else:
        run_compare_cli(args.csv, args.model)
