"""Macro-economic features.

Calculates macro indicators:
- VIX (Volatility Index)
- Interest rates (using TLT as proxy)
- Inflation/Safe-haven (using GLD as proxy)
- SPY returns for broad market trend
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import START_DATE, END_DATE

MACRO_TICKERS = {
    "VIX": "^VIX",
    "TLT": "TLT",   # Treasury yields proxy
    "GLD": "GLD",   # Safe haven / Inflation proxy
    "SPY": "SPY"    # Market proxy
}

def load_macro_data(start_date=START_DATE, end_date=END_DATE) -> pd.DataFrame:
    """Download macro indicators from yfinance."""
    data = yf.download(list(MACRO_TICKERS.values()), start=start_date, end=end_date, progress=False)
    
    # Check if data is MultiIndex (if multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].copy()
    else:
        closes = pd.DataFrame(data["Close"])
    
    # Rename columns to our macro names
    rename_map = {v: k for k, v in MACRO_TICKERS.items()}
    closes = closes.rename(columns=rename_map)
    closes = closes.ffill()
    return closes

def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived macro features."""
    feats = pd.DataFrame(index=df.index)
    
    # VIX level and its momentum
    if "VIX" in df.columns:
        feats["vix_level"] = df["VIX"]
        feats["vix_1m_change"] = df["VIX"].pct_change(21)
    
    # TLT (Treasury) momentum - proxy for interest rate changes
    if "TLT" in df.columns:
        feats["tlt_1m_ret"] = df["TLT"].pct_change(21)
        feats["tlt_vol_20"] = df["TLT"].pct_change(1).rolling(20).std()
        
    # GLD (Gold) momentum - proxy for inflation/fear
    if "GLD" in df.columns:
        feats["gld_1m_ret"] = df["GLD"].pct_change(21)
        
    # SPY (Market) momentum
    if "SPY" in df.columns:
        feats["spy_1m_ret"] = df["SPY"].pct_change(21)
        feats["spy_vol_20"] = df["SPY"].pct_change(1).rolling(20).std()
        
    return feats

def build_macro_feature_matrix(start_date=START_DATE, end_date=END_DATE) -> pd.DataFrame:
    """Download and build the complete macro feature matrix."""
    raw_macro = load_macro_data(start_date, end_date)
    macro_feats = compute_macro_features(raw_macro)
    macro_feats = macro_feats.replace([np.inf, -np.inf], np.nan)
    macro_feats = macro_feats.ffill().bfill()
    return macro_feats

if __name__ == "__main__":
    macro_df = build_macro_feature_matrix()
    print("Macro features shape:", macro_df.shape)
    print(macro_df.tail())
