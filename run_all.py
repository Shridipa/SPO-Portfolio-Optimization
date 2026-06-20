#!/usr/bin/env python3
"""Full pipeline orchestrator for the SPO Portfolio Optimization project.

Runs the entire project pipeline:
    1. Build feature matrix
    2. Run baseline experiments (EqualWeight, MeanVariance, Linear+MVO, XGBoost+MVO, LSTM+MVO)
    3. Run SPO experiments (MLP and LSTM variants)
    4. Run SPO + DRO experiments
    5. Run regime detection
    6. Produce comparison table
    7. Generate visualizations

Usage:
    python run_all.py              # full pipeline (takes time)
    python run_all.py --quick      # quick test (few epochs)
    python run_all.py --skip-spo   # skip SPO experiments
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TICKERS, N_ASSETS, SPO_LR, SPO_EPOCHS
from data.feature_pipeline import build_feature_matrix, build_price_matrix, build_return_matrix
from backtest.metrics import metrics_to_dataframe

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)


def step_1_load_data():
    """Step 1: Load and prepare all data."""
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    prices = build_price_matrix()
    returns = build_return_matrix(prices, periods=1)
    features = build_feature_matrix()

    # Align
    common_dates = prices.index.intersection(returns.index)
    if features is not None and len(features) > 0:
        feature_dates = features.index.get_level_values(0).unique()
        common_dates = common_dates.intersection(feature_dates)

    prices = prices.loc[common_dates]
    returns = returns.loc[common_dates]

    print(f"Prices: {prices.shape} [{prices.index[0]:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d}]")
    print(f"Returns: {returns.shape}")
    if features is not None:
        print(f"Features: {features.shape} ({len(features.columns) * N_ASSETS} total)")

    return prices, returns, features


def step_2_baseline(prices, returns, features, quick=False):
    """Step 2: Run baseline experiments."""
    from experiments.baseline import run_baseline_experiment, print_results_table

    print("\n" + "=" * 80)
    print("STEP 2: BASELINE EXPERIMENTS")
    print("=" * 80)

    results = run_baseline_experiment(
        prices, returns, features,
        train_window=252 * 3,  # shorter for speed
        transaction_cost=0.001,
        verbose=True,
    )

    df = print_results_table(results)
    df.to_csv(RESULTS_DIR / "00_baseline_results.csv")
    return results


def step_3_spo(prices, returns, features, quick=False, use_lstm=False):
    """Step 3: Run SPO experiments."""
    from experiments.spo import run_spo_experiment

    print("\n" + "=" * 80)
    print(f"STEP 3: SPO EXPERIMENT ({'LSTM' if use_lstm else 'MLP'})")
    print("=" * 80)

    epochs = 5 if quick else SPO_EPOCHS
    suffix = "_lstm" if use_lstm else "_mlp"

    metrics = run_spo_experiment(
        prices, returns, features,
        use_lstm=use_lstm,
        predict_cov=False,
        epochs=epochs,
        lr=SPO_LR,
        verbose=True,
    )

    pd.Series(metrics).to_csv(RESULTS_DIR / f"01_spo{suffix}_results.csv")
    return metrics


def step_4_spo_dro(prices, returns, features, quick=False):
    """Step 4: Run SPO + DRO experiments."""
    from experiments.spo_dro import run_spo_dro_experiment

    print("\n" + "=" * 80)
    print("STEP 4: SPO + DRO EXPERIMENT")
    print("=" * 80)

    epochs = 5 if quick else SPO_EPOCHS

    metrics = run_spo_dro_experiment(
        prices, returns, features,
        epochs=epochs,
        lr=SPO_LR,
        verbose=True,
    )

    pd.Series(metrics).to_csv(RESULTS_DIR / "02_spo_dro_results.csv")
    return metrics


def step_5_regime(prices, returns, features):
    """Step 5: Run regime detection."""
    from experiments.spo_dro import run_regime_spo_experiment

    print("\n" + "=" * 80)
    print("STEP 5: REGIME DETECTION")
    print("=" * 80)

    summary = run_regime_spo_experiment(prices, returns, features, verbose=True)
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "03_regime_summary.csv")
    return summary


def step_6_comparison(all_metrics):
    """Step 6: Produce comparison table."""
    print("\n" + "=" * 80)
    print("STEP 6: COMPARISON TABLE")
    print("=" * 80)

    rows = []

    # Baseline models
    if "baseline" in all_metrics:
        for model_name, metrics in all_metrics["baseline"].items():
            row = {"Model": model_name}
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating)):
                    row[k] = v
            rows.append(row)

    # SPO models
    for key in ["spo_mlp", "spo_lstm"]:
        if key in all_metrics:
            row = {"Model": key.upper()}
            for k, v in all_metrics[key].items():
                if isinstance(v, (int, float, np.floating)):
                    row[k] = v
            rows.append(row)

    # SPO + DRO
    if "spo_dro" in all_metrics:
        row = {"Model": "SPO+DRO"}
        for k, v in all_metrics["spo_dro"].items():
            if isinstance(v, (int, float, np.floating)):
                row[k] = v
        rows.append(row)

    if not rows:
        print("No results to compare.")
        return None

    comparison = pd.DataFrame(rows).set_index("Model")
    key_cols = [c for c in ["sharpe", "ann_return", "ann_vol", "max_drawdown", "turnover"]
                if c in comparison.columns]
    display = comparison[key_cols].round(4)
    display = display.sort_values("sharpe", ascending=False)

    print("\nFinal Comparison:")
    print("-" * 80)
    print(display.to_string())
    print("-" * 80)

    comparison.to_csv(RESULTS_DIR / "99_final_comparison.csv")
    print(f"Comparison table saved to {RESULTS_DIR / '99_final_comparison.csv'}")
    return comparison


def generate_visualizations(prices, returns, all_metrics):
    """Generate plots for the results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        print("\n=== Generating Visualizations ===")

        # 1. Cumulative return comparison
        plt.figure(figsize=(14, 6))
        if "baseline" in all_metrics:
            for model_name, metrics in all_metrics["baseline"].items():
                if "ann_return" in metrics:
                    # Simulated cumulative return from annualized return
                    cum = np.cumprod(1 + np.ones(100) * metrics.get("ann_return", 0) / 252)
                    plt.plot(cum, label=model_name)
        plt.title("Portfolio Performance Comparison")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "comparison.png", dpi=150)
        print(f"Saved {RESULTS_DIR / 'comparison.png'}")

        # 2. Sharpe ratio bar chart
        plt.figure(figsize=(12, 5))
        model_names = []
        sharpes = []
        for group_name, group in all_metrics.items():
            if isinstance(group, dict):
                for model_name, metrics in group.items():
                    if isinstance(metrics, dict) and "sharpe" in metrics:
                        model_names.append(f"{model_name}")
                        sharpes.append(metrics["sharpe"])
        if sharpes:
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sharpes)))
            plt.barh(model_names, sharpes, color=colors)
            plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Target Sharpe=1.0')
            plt.title("Sharpe Ratio Comparison")
            plt.xlabel("Sharpe Ratio")
            plt.legend()
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "sharpe_comparison.png", dpi=150)
            print(f"Saved {RESULTS_DIR / 'sharpe_comparison.png'}")

        plt.close('all')
        print("Visualizations complete.")
    except Exception as e:
        print(f"Visualization error (non-fatal): {e}")


def main():
    parser = argparse.ArgumentParser(description="SPO Portfolio — Full Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick test (few epochs)")
    parser.add_argument("--skip-spo", action="store_true", help="Skip SPO experiments")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline experiments")
    parser.add_argument("--skip-dro", action="store_true", help="Skip SPO+DRO experiments")
    parser.add_argument("--skip-regime", action="store_true", help="Skip regime detection")
    parser.add_argument("--skip-plots", action="store_true", help="Skip visualizations")
    args = parser.parse_args()

    print("=" * 80)
    print("  SPO PORTFOLIO OPTIMIZATION — FULL PIPELINE")
    print("=" * 80)
    print(f"  Quick mode: {args.quick}")
    print(f"  Assets: {', '.join(TICKERS[:5])}... ({N_ASSETS} total)")
    print("=" * 80)

    # Step 1: Load data
    prices, returns, features = step_1_load_data()

    all_metrics = {}

    # Step 2: Baseline
    if not args.skip_baseline:
        all_metrics["baseline"] = step_2_baseline(prices, returns, features, args.quick)

    # Step 3: SPO
    if not args.skip_spo:
        all_metrics["spo_mlp"] = step_3_spo(prices, returns, features, args.quick, use_lstm=False)
        if args.quick:
            # Only MLP in quick mode
            pass
        else:
            all_metrics["spo_lstm"] = step_3_spo(prices, returns, features, args.quick, use_lstm=True)

    # Step 4: SPO + DRO
    if not args.skip_dro and not args.skip_spo:
        all_metrics["spo_dro"] = step_4_spo_dro(prices, returns, features, args.quick)

    # Step 5: Regime
    if not args.skip_regime:
        all_metrics["regime"] = step_5_regime(prices, returns, features)

    # Step 6: Comparison
    comparison = step_6_comparison(all_metrics)

    # Visualizations
    if not args.skip_plots:
        generate_visualizations(prices, returns, all_metrics)

    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  All results saved to: {RESULTS_DIR.resolve()}")
    print(f"  Saved models: {MODELS_DIR.resolve()}")
    print("=" * 80)

    return comparison


if __name__ == "__main__":
    main()
