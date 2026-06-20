"""Download market data using yfinance and save to CSV files.

Provides a function `download_data(tickers, start, end, interval='1d')` that
fetches historical OHLCV data for the given list of tickers and stores each
ticker's data under ``data/raw/<ticker>.csv``.
"""
import os
from pathlib import Path
import yfinance as yf
import pandas as pd

RAW_DATA_DIR = Path(__file__).parent / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_data(tickers, start="2010-01-01", end=None, interval="1d"):
    """Download OHLCV data for ``tickers``.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str or None
        End date. If ``None`` uses current date.
    interval : str
        Data interval (e.g., ``'1d'``, ``'1h'``).
    """
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            continue
        df.reset_index(inplace=True)
        out_path = RAW_DATA_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {ticker} to {out_path}")

if __name__ == "__main__":
    DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "AAPL", "MSFT", "NVDA", "JPM", "XLE", "XLV"]
    download_data(DEFAULT_TICKERS)
