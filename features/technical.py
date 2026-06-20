"""Technical feature engineering utilities.

This module provides functions to compute common technical indicators
required for the portfolio prediction pipeline. The implementations use
pandas and numpy and are intentionally lightweight – they return the
computed series so that callers can concatenate them to a feature DataFrame.
"""
import pandas as pd
import numpy as np

def compute_returns(df: pd.DataFrame, periods: list[int] = [1, 5, 20]) -> pd.DataFrame:
    """Compute percentage returns over given look‑back periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``Close`` column.
    periods : list[int]
        List of integer periods (in days) for which to compute returns.
    """
    returns = {}
    for p in periods:
        returns[f"ret_{p}d"] = df['Close'].pct_change(p)
    return pd.DataFrame(returns)

def sma_ratio(df: pd.DataFrame, windows: list[int] = [5, 20]) -> pd.DataFrame:
    """Simple moving‑average ratios (price / SMA)."""
    ratios = {}
    for w in windows:
        sma = df['Close'].rolling(w).mean()
        ratios[f"sma_ratio_{w}d"] = df['Close'] / sma
    return pd.DataFrame(ratios)

def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index.

    Returns a pandas Series aligned with ``df``.
    """
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence components.
    Returns DataFrame with ``macd`` and ``macd_signal`` columns.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line})

def volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualised volatility (standard deviation of log returns)."""
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    vol = log_ret.rolling(window).std() * np.sqrt(252)
    return vol

def momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple momentum as price difference over ``window`` days."""
    return df['Close'].diff(window)

def rolling_skewness(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df['Close'].rolling(window).skew()

def rolling_kurtosis(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df['Close'].rolling(window).kurt()
