"""Training script for the end-to-end SPO neural portfolio model.

Usage:
    python train.py --mode spo --epochs 100 --lr 0.001
    python train.py --mode baseline
    python train.py --mode all
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TICKERS, SPO_LR, SPO_EPOCHS
from data.feature_pipeline import build_feature_matrix, build_price_matrix, build_return_matrix
from experiments.baseline import run_baseline_experiment, print_results_table
from experiments.spo import run_spo_experiment
from experiments.spo_dro import run_spo_dro_experiment, run_regime_spo_experiment

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load preprocessed data."""
    print("\n=== Loading Data ===")
    prices = build_price_matrix()
    returns = build_return_matrix(prices, periods=1)
    features = build_feature_matrix()

    # Align all data
    common_dates = prices.index.intersection(returns.index)
    if features is not None and len(features) > 0:
        feature_dates = features.index.get_level_values(0).unique()
        common_dates = common_dates.intersection(feature_dates)

    prices = prices.loc[common_dates]
    returns = returns.loc[common_dates]

    print(f"Data loaded: {len(prices)} days, {prices.shape[1]} assets")
    print(f"Date range: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")
    return prices, returns, features


def run_baseline(prices, returns, features, verbose=True):
    """Run all baseline models."""
    print("\n" + "=" * 80)
    print("RUNNING BASELINE EXPERIMENTS")
    print("=" * 80)

    results = run_baseline_experiment(
        prices, returns, features,
        train_window=252 * 5,
        transaction_cost=0.001,
        verbose=verbose,
    )

    df = print_results_table(results)
    df.to_csv(RESULTS_DIR / "baseline_results.csv")
    print(f"Results saved to {RESULTS_DIR / 'baseline_results.csv'}")
    return results


def run_spo(prices, returns, features, use_lstm=False, predict_cov=False, verbose=True):
    """Run SPO experiment."""
    print("\n" + "=" * 80)
    print(f"RUNNING SPO EXPERIMENT ({'LSTM' if use_lstm else 'MLP'})")
    print("=" * 80)

    suffix = "_lstm" if use_lstm else "_mlp"
    metrics = run_spo_experiment(
        prices, returns, features,
        use_lstm=use_lstm,
        predict_cov=predict_cov,
        epochs=SPO_EPOCHS,
        lr=SPO_LR,
        verbose=verbose,
    )

    pd.Series(metrics).to_csv(RESULTS_DIR / f"spo{suffix}_results.csv")
    return metrics


def run_spo_dro(prices, returns, features, verbose=True):
    """Run SPO + DRO experiment."""
    print("\n" + "=" * 80)
    print("RUNNING SPO + DRO EXPERIMENT")
    print("=" * 80)

    metrics = run_spo_dro_experiment(prices, returns, features, verbose=verbose)
    pd.Series(metrics).to_csv(RESULTS_DIR / "spo_dro_results.csv")
    return metrics


def run_regime(prices, returns, features, verbose=True):
    """Run regime detection experiment."""
    print("\n" + "=" * 80)
    print("RUNNING REGIME DETECTION EXPERIMENT")
    print("=" * 80)

    summary = run_regime_spo_experiment(prices, returns, features, verbose=verbose)
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "regime_summary.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SPO Portfolio Training")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "spo", "spo_dro", "regime", "all"],
                        help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=SPO_EPOCHS)
    parser.add_argument("--lr", type=float, default=SPO_LR)
    parser.add_argument("--use_lstm", action="store_true")
    parser.add_argument("--predict_cov", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer epochs for quick test")
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 5)

    prices, returns, features = load_data()

    all_metrics = {}

    if args.mode in ("baseline", "all"):
        bl_results = run_baseline(prices, returns, features)
        all_metrics["baseline"] = bl_results

    if args.mode in ("spo", "all"):
        spo_metrics = run_spo(prices, returns, features,
                              use_lstm=args.use_lstm,
                              predict_cov=args.predict_cov)
        all_metrics["spo"] = spo_metrics

    if args.mode in ("spo_dro", "all"):
        dro_metrics = run_spo_dro(prices, returns, features)
        all_metrics["spo_dro"] = dro_metrics

    if args.mode in ("regime", "all"):
        regime_info = run_regime(prices, returns, features)
        all_metrics["regime"] = regime_info

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"Results saved to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
