"""Baseline experiment runner.

Compares:
1. Equal-weight portfolio
2. Mean-Variance (historical mean + shrinkage covariance)
3. Linear + MVO (predict-then-optimize)
4. XGBoost + MVO
5. LSTM + MVO
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TICKERS, LAMBDA_REG, MAX_WEIGHT
from data.feature_pipeline import build_price_matrix, build_return_matrix
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, metrics_to_dataframe
from models.linear import LinearReturnModel
from models.xgboost_model import XGBoostReturnModel
from models.lstm import LSTMReturnModel


def run_baseline_experiment(prices: pd.DataFrame, returns: pd.DataFrame,
                            feature_df: pd.DataFrame = None,
                            train_window: int = 252 * 5,
                            transaction_cost: float = 0.001,
                            verbose: bool = True) -> dict:
    """Run all baseline models and return metrics.

    Args:
        prices: (T, N) DataFrame of closing prices
        returns: (T, N) DataFrame of forward returns
        feature_df: (T, N*F) or (T*N, F) feature matrix (multi-index)
        train_window: lookback window in days
        transaction_cost: cost per unit weight change
        verbose: print progress

    Returns:
        dict of {model_name: {metric: value}}
    """
    results = {}
    n_assets = prices.shape[1]

    # 1. Equal-weight baseline
    if verbose:
        print("\n=== Running Equal-Weight Baseline ===")
    ew_returns = returns.mean(axis=1).values
    ew_metrics = compute_metrics(ew_returns, freq=252)
    results["EqualWeight"] = ew_metrics

    # 2. Mean-Variance (historical)
    if verbose:
        print("\n=== Running Mean-Variance (Historical) ===")
    bt_mv = BacktestEngine(prices, returns,
                           train_window=train_window,
                           transaction_cost=transaction_cost)
    bt_mv.run(mu_predictor=None, verbose=verbose)
    mv_metrics = compute_metrics(bt_mv.portfolio_returns, bt_mv.weights_history, freq=252)
    results["MeanVariance"] = mv_metrics

    # 3. Linear + MVO
    if verbose:
        print("\n=== Running Linear + MVO ===")
    if feature_df is not None:
        # Flatten features for the linear model
        if isinstance(feature_df.index, pd.MultiIndex):
            feature_flat = feature_df.unstack("ticker")
            feature_flat.columns = [f"{a}_{b}" for a, b in feature_flat.columns]
        else:
            feature_flat = feature_df

        # Align dates
        common_dates = feature_flat.index.intersection(returns.index)
        X = feature_flat.loc[common_dates]
        y = returns.loc[common_dates]

        def make_predictor(model, X_all):
            preds = model.predict(X_all)
            pred_df = pd.DataFrame(preds, index=X_all.index)
            def predictor(date):
                if date in pred_df.index:
                    val = pred_df.loc[date].values
                    if pd.isna(val).any():
                        return None
                    return val
                return None
            return predictor

        # Train linear model on early data, walk-forward predict
        split = len(X) // 2
        model = LinearReturnModel()
        model.fit(X.iloc[:split], y.iloc[:split])

        bt_lin = BacktestEngine(prices, returns,
                                train_window=train_window,
                                transaction_cost=transaction_cost)
        bt_lin.run(mu_predictor=make_predictor(model, X), verbose=verbose)
        lin_metrics = compute_metrics(bt_lin.portfolio_returns, bt_lin.weights_history, freq=252)
        results["Linear+MVO"] = lin_metrics

        # 4. XGBoost + MVO
        if verbose:
            print("\n=== Running XGBoost + MVO ===")
        xgb_model = XGBoostReturnModel(n_estimators=200)
        xgb_model.fit(X.iloc[:split], y.iloc[:split])

        bt_xgb = BacktestEngine(prices, returns,
                                train_window=train_window,
                                transaction_cost=transaction_cost)
        bt_xgb.run(mu_predictor=make_predictor(xgb_model, X), verbose=verbose)
        xgb_metrics = compute_metrics(bt_xgb.portfolio_returns, bt_xgb.weights_history, freq=252)
        results["XGBoost+MVO"] = xgb_metrics

        # 5. LSTM + MVO
        if verbose:
            print("\n=== Running LSTM + MVO ===")
        try:
            n_features = X.shape[1]
            lstm_model = LSTMReturnModel(n_features, n_assets, seq_len=60)
            lstm_model.fit(X.iloc[:split], y.iloc[:split], epochs=50, verbose=False)

            bt_lstm = BacktestEngine(prices, returns,
                                     train_window=train_window,
                                     transaction_cost=transaction_cost)
            bt_lstm.run(mu_predictor=make_predictor(lstm_model, X), verbose=verbose)
            lstm_metrics = compute_metrics(bt_lstm.portfolio_returns, bt_lstm.weights_history, freq=252)
            results["LSTM+MVO"] = lstm_metrics
        except Exception as e:
            if verbose:
                print(f"  LSTM failed: {e}")
            results["LSTM+MVO"] = {"sharpe": 0, "ann_return": 0}

    return results


def print_results_table(results: dict):
    """Pretty-print results as a table."""
    df = metrics_to_dataframe(results)
    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    key_metrics = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover", "win_rate"]
    display = df[[c for c in key_metrics if c in df.columns]]
    print(display.to_string())
    print("=" * 80)
    return df
