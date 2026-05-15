import os


def pytest_configure():
    os.environ.setdefault("BOT_MODE", "backtest")
    os.environ.setdefault("BROKER", "paper")
    os.environ.setdefault("ENABLE_REAL_TRADING", "false")
