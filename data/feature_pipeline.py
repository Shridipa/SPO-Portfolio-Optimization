"""Feature engineering pipeline: loads raw data, computes features, aligns across tickers."""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TICKERS, LOOKBACK_DAYS
from features.technical import (
    compute_returns, sma_ratio, rsi, macd,
    volatility, momentum, rolling_skewness, rolling_kurtosis
)

RAW_DIR = Path(__file__).parent / "raw"


def load_ticker(ticker: str) -> pd.DataFrame:
    """Load a single ticker's raw CSV."""
    path = RAW_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"] if "Date" in open(path).readline() else ["Datetime"])
    date_col = [c for c in df.columns if "Date" in c or "date" in c or "Datetime" in c][0]
    df = df.rename(columns={date_col: "Date"})
    df = df.sort_values("Date").set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Drop duplicate indices
    df = df[~df.index.duplicated(keep="first")]
    # Keep only OHLCV columns and standardise names to Title Case
    col_map = {c: c.title() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume", "adj close")}
    df = df[list(col_map.keys())].rename(columns=col_map)
    return df


def build_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical features for a single ticker's OHLCV DataFrame."""
    feats = []
    # Returns
    feats.append(compute_returns(df, periods=[1, 5, 20]))
    # SMA ratios
    feats.append(sma_ratio(df, windows=[5, 20, 50]))
    # RSI
    rsi_series = rsi(df, window=14)
    feats.append(pd.DataFrame({"rsi_14": rsi_series}))
    # MACD
    feats.append(macd(df, fast=12, slow=26, signal=9))
    # Volatility
    vol_series = volatility(df, window=20)
    feats.append(pd.DataFrame({"volatility_20": vol_series}))
    # Momentum
    mom_series = momentum(df, window=20)
    feats.append(pd.DataFrame({"momentum_20": mom_series}))
    # Rolling skewness / kurtosis
    feats.append(pd.DataFrame({"skew_20": rolling_skewness(df, window=20)}))
    feats.append(pd.DataFrame({"kurt_20": rolling_kurtosis(df, window=20)}))
    # Price-based
    close_col = "Close" if "Close" in df.columns else "close"
    vol_col = "Volume" if "Volume" in df.columns else "volume"
    feats.append(pd.DataFrame({
        "close": df[close_col],
        "volume": df.get(vol_col, pd.Series(index=df.index, dtype=float))
    }))
    result = pd.concat(feats, axis=1)
    return result


def build_feature_matrix(start: str = "2010-01-01", end: str = "2026-06-20") -> pd.DataFrame:
    """Build a multi-index feature matrix (date, ticker) -> features."""
    all_dfs = []
    for ticker in TICKERS:
        raw = load_ticker(ticker)
        raw = raw.loc[start:end]
        feats = build_features_for_ticker(raw)
        feats["ticker"] = ticker
        all_dfs.append(feats.reset_index())
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.set_index(["Date", "ticker"]).sort_index()
    # Drop infinite and extreme outliers
    combined = combined.replace([np.inf, -np.inf], np.nan)
    # Fill remaining NaNs with forward fill then backward fill
    combined = combined.groupby("ticker").apply(lambda g: g.ffill().bfill())
    combined = combined.droplevel(0)  # remove extra index from groupby
    return combined


def build_price_matrix(tickers=None, start="2010-01-01", end="2026-06-20"):
    """Build a T x N DataFrame of closing prices."""
    if tickers is None:
        tickers = TICKERS
    prices = []
    for t in tickers:
        raw = load_ticker(t)
        raw = raw.loc[start:end]
        close_col = "Close" if "Close" in raw.columns else "close"
        prices.append(raw[close_col].rename(t))
    return pd.concat(prices, axis=1)


def build_return_matrix(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Build a T x N DataFrame of forward returns."""
    return prices.pct_change(periods).shift(-periods)


if __name__ == "__main__":
    # Quick test
    feat_matrix = build_feature_matrix()
    print(f"Feature matrix shape: {feat_matrix.shape}")
    print(f"Columns: {list(feat_matrix.columns)}")
    prices = build_price_matrix()
    print(f"Price matrix shape: {prices.shape}")
    feat_matrix.to_csv(Path(__file__).parent / "features.csv")
    prices.to_csv(Path(__file__).parent / "prices.csv")
    print("Saved features.csv and prices.csv")
