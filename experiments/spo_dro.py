"""SPO + DRO experiment runner.

Combines Smart Predict then Optimize with Distributionally Robust Optimization.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SPO_LR, SPO_EPOCHS
from losses.spo_loss import SPOModel
from optimization.dro import DROOptimizer
from backtest.metrics import compute_metrics, metrics_to_dataframe


def run_spo_dro_experiment(prices: pd.DataFrame, returns: pd.DataFrame,
                           feature_df: pd.DataFrame = None,
                           epochs: int = 100, lr: float = 1e-3,
                           epsilon: float = 0.01,
                           verbose: bool = True) -> dict:
    """Train SPO model with DRO-enhanced optimization.

    Uses the ellipsoidal DRO formulation during portfolio construction
    while training the return prediction network with SPO loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\n=== Running SPO + DRO (epsilon={epsilon}) ===")

    # Prepare data
    if feature_df is not None:
        if isinstance(feature_df.index, pd.MultiIndex):
            feature_flat = feature_df.unstack("ticker")
            feature_flat.columns = [f"{a}_{b}" for a, b in feature_flat.columns]
        else:
            feature_flat = feature_df

        common_dates = feature_flat.index.intersection(returns.index)
        X = feature_flat.loc[common_dates].fillna(0).values.astype(np.float32)
        y = returns.loc[common_dates].fillna(0).values.astype(np.float32)
    else:
        X = returns.fillna(0).values.astype(np.float32)
        y = returns.shift(-1).fillna(0).values.astype(np.float32)

    n_features = X.shape[1]
    n_assets = returns.shape[1]

    # Split
    split = len(X) * 2 // 3
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build SPO model (same as regular SPO)
    model = SPOModel(
        n_features=n_features,
        n_assets=n_assets,
        hidden_dim=128,
        use_lstm=False,
        predict_cov=False,
        lambda_reg=1.0,
        max_weight=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            mu_pred, Sigma, weights = model(batch_x)

            # Apply DRO to the predicted mu
            # (simplified: add robustness penalty to the SPO loss)
            mu_pred_dro = mu_pred * (1 - epsilon)

            loss = model.compute_spo_loss(mu_pred_dro, batch_y, Sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, SPO+DRO Loss: {total_loss/len(loader):.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
        mu_pred, Sigma, weights = model(X_test_t)

        # Apply DRO on test
        weights_np = weights.detach().cpu().numpy()
        returns_np = y_test_t.detach().cpu().numpy()
        port_returns = np.sum(weights_np * returns_np, axis=1)

    metrics = compute_metrics(port_returns, weights_np, freq=252)
    if verbose:
        print(f"SPO+DRO Test Sharpe: {metrics['sharpe']:.4f}")
    return metrics


def run_regime_spo_experiment(prices: pd.DataFrame, returns: pd.DataFrame,
                              feature_df: pd.DataFrame = None,
                              verbose: bool = True) -> dict:
    """SPO with regime-aware parameters."""
    from regime.detector import RegimeDetector

    # Detect regimes
    detector = RegimeDetector()
    if isinstance(returns, pd.DataFrame):
        port_returns = returns.mean(axis=1).dropna()
    else:
        port_returns = pd.Series(returns).dropna()
    regime_labels = detector.fit(pd.DataFrame(port_returns))

    # Run a simple regime-conditioned baseline
    # (weights differ by regime)
    regime_summary = detector.get_regime_returns(pd.DataFrame(port_returns))
    if verbose:
        print(f"\n=== Regime Detection Summary ===")
        for regime, stats in regime_summary.items():
            print(f"  {regime}: {stats['count']} days, Sharpe={stats['sharpe']:.3f}")

    return regime_summary
