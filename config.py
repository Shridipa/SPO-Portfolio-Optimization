"""Global configuration for the SPO Portfolio Optimization project."""
from dataclasses import dataclass, field
from typing import List

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "JPM", "XLE", "XLV"]
N_ASSETS = len(TICKERS)

# Dates
START_DATE = "2010-01-01"
END_DATE = "2026-06-20"
TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"

# Walk-forward
TRAIN_WINDOW_YEARS = 5
REBALANCE_FREQ = "monthly"  # monthly or weekly

# Features
LOOKBACK_DAYS = 60  # for LSTM/Transformer
TECHNICAL_PERIODS = [1, 5, 20]

# Optimization
LAMBDA_REG = 1.0  # risk aversion
MAX_WEIGHT = 0.2
RISK_FREE_RATE = 0.0

# Model paths
MODEL_DIR = "saved_models"

# SPO
SPO_LR = 1e-3
SPO_EPOCHS = 100
SPO_BATCH_SIZE = 64

# Regime
REGIME_STATES = ["Bull", "Bear", "Sideways", "HighVol"]
N_REGIMES = len(REGIME_STATES)


@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: TICKERS)
    start_date: str = START_DATE
    end_date: str = END_DATE
    train_end: str = TRAIN_END
    val_end: str = VAL_END
    train_window_years: int = TRAIN_WINDOW_YEARS
    rebalance_freq: str = REBALANCE_FREQ
    lookback_days: int = LOOKBACK_DAYS
    lambda_reg: float = LAMBDA_REG
    max_weight: float = MAX_WEIGHT
    risk_free_rate: float = RISK_FREE_RATE
    spo_lr: float = SPO_LR
    spo_epochs: int = SPO_EPOCHS
    spo_batch_size: int = SPO_BATCH_SIZE
