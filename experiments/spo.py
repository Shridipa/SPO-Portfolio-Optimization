"""SPO (Smart Predict then Optimize) experiment runner.

Trains an end-to-end differentiable portfolio model using decision loss.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TICKERS, N_ASSETS, SPO_LR, SPO_EPOCHS
from losses.spo_loss import SPOModel
from backtest.metrics import compute_metrics, metrics_to_dataframe


def run_spo_experiment(prices: pd.DataFrame, returns: pd.DataFrame,
                       feature_df: pd.DataFrame = None,
                       n_features: int = 60, hidden_dim: int = 128,
                       epochs: int = SPO_EPOCHS, lr: float = SPO_LR,
                       use_lstm: bool = False, predict_cov: bool = False,
                       lambda_reg: float = 1.0, max_weight: float = 0.2,
                       verbose: bool = True) -> dict:
    """Train and evaluate an end-to-end SPO model.

    Args:
        prices: (T, N) closing prices
        returns: (T, N) forward returns
        feature_df: (T, N*F) features (if None, use returns directly)
        n_features: number of input features
        epochs: training epochs
        lr: learning rate
        use_lstm: use LSTM encoder if True, else MLP
        predict_cov: predict covariance matrix jointly
        lambda_reg: risk aversion
        max_weight: max weight per asset
        verbose: print progress

    Returns:
        dict of {metric: value}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"\n=== Running SPO ({'LSTM' if use_lstm else 'MLP'}) ===")
        print(f"Device: {device}")

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
        # Use the returns themselves as features (for simple test)
        X = returns.fillna(0).values.astype(np.float32)
        y = returns.shift(-1).fillna(0).values.astype(np.float32)

    n_features_actual = X.shape[1]
    n_assets = returns.shape[1]

    # Split
    split = len(X) * 2 // 3
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = SPOModel(
        n_features=n_features_actual,
        n_assets=n_assets,
        hidden_dim=hidden_dim,
        use_lstm=use_lstm,
        seq_len=60,
        predict_cov=predict_cov,
        lambda_reg=lambda_reg,
        max_weight=max_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    spo_loss_fn = model.compute_spo_loss

    # Training loop
    if use_lstm:
        # Build sequences
        seq_len = 60
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X_train)):
            X_seq.append(X_train[i - seq_len:i])
            y_seq.append(y_train[i])
        X_train_t = torch.tensor(np.array(X_seq), dtype=torch.float32, device=device)
        y_train_t = torch.tensor(np.array(y_seq), dtype=torch.float32, device=device)
    else:
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    best_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            mu_pred, Sigma, weights = model(batch_x)
            loss = spo_loss_fn(mu_pred, batch_y, Sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, SPO Loss: {avg_loss:.6f}")

    if verbose:
        print(f"Training complete. Best SPO Loss: {best_loss:.6f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        if use_lstm:
            X_seq_test, y_seq_test = [], []
            for i in range(seq_len, len(X_test)):
                X_seq_test.append(X_test[i - seq_len:i])
                y_seq_test.append(y_test[i])
            if len(X_seq_test) > 0:
                X_test_t = torch.tensor(np.array(X_seq_test), dtype=torch.float32, device=device)
                y_test_t = torch.tensor(np.array(y_seq_test), dtype=torch.float32, device=device)
                mu_pred, Sigma, weights = model(X_test_t)
                test_loss = spo_loss_fn(mu_pred, y_test_t, Sigma).item()

                # Compute portfolio returns on test set
                weights_np = weights.detach().cpu().numpy()
                returns_np = y_test_t.detach().cpu().numpy()
                port_returns = np.sum(weights_np * returns_np, axis=1)
            else:
                port_returns = np.array([0.0])
                test_loss = 0.0
        else:
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
            mu_pred, Sigma, weights = model(X_test_t)
            test_loss = spo_loss_fn(mu_pred, y_test_t, Sigma).item()

            weights_np = weights.detach().cpu().numpy()
            returns_np = y_test_t.detach().cpu().numpy()
            port_returns = np.sum(weights_np * returns_np, axis=1)

    metrics = compute_metrics(port_returns, weights_np, freq=252)
    metrics["test_spo_loss"] = test_loss

    if verbose:
        print(f"Test SPO Loss: {test_loss:.6f}")
        print(f"Test Sharpe: {metrics['sharpe']:.4f}")
        print(f"Test Ann Return: {metrics['ann_return']:.4f}")

    return metrics
