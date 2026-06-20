"""SPO (Smart Predict then Optimize) loss and end-to-end model.

Key idea: Instead of minimizing prediction error (MSE), minimize
decision regret — the difference between optimal portfolio value
(true returns) and portfolio value achieved using predicted returns.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimization.differentiable_layer import DifferentiableMarkowitzLayer


class CovarianceNetwork(nn.Module):
    """Neural network that predicts a positive-definite covariance matrix.

    Outputs Cholesky factor L such that Sigma = L @ L^T.
    """

    def __init__(self, n_features: int, n_assets: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.n_assets = n_assets
        self.n_cholesky = n_assets * (n_assets + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_cholesky),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Sigma: (batch, n_assets, n_assets) positive semidefinite."""
        raw = self.net(x)  # (batch, n_cholesky)
        batch_size = x.shape[0]

        L = torch.zeros(batch_size, self.n_assets, self.n_assets, device=x.device, dtype=x.dtype)
        idx = 0
        for i in range(self.n_assets):
            length = self.n_assets - i
            L[:, i, i:] = raw[:, idx:idx + length]
            idx += length

        # Ensure positive diagonal via softplus
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_positive = nn.functional.softplus(diag)
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag_positive)

        # Sigma = L @ L^T (guarantees PSD)
        Sigma = L @ L.transpose(-2, -1)
        return Sigma


class SPOLoss(nn.Module):
    """SPO decision loss.

    L(w_pred, mu_true) = max_w (mu_true @ w) - (mu_true @ w_pred)

    This is the "regret": the value gap between the optimal portfolio
    (using true returns) and the portfolio built with predicted returns.
    """

    def __init__(self, n_assets: int, lambda_reg: float = 1.0, max_weight: float = 0.2):
        super().__init__()
        self.n_assets = n_assets
        self.opt_layer = DifferentiableMarkowitzLayer(n_assets, lambda_reg, max_weight)

    def forward(self, mu_pred: torch.Tensor, mu_true: torch.Tensor,
                Sigma: torch.Tensor = None) -> torch.Tensor:
        """Compute SPO loss.

        Args:
            mu_pred: (batch, n_assets) predicted returns
            mu_true: (batch, n_assets) true returns
            Sigma: (batch, n_assets, n_assets) covariance (if None, use identity)

        Returns:
            loss: scalar
        """
        batch_size = mu_pred.shape[0]

        if Sigma is None:
            eye = torch.eye(self.n_assets, device=mu_pred.device, dtype=mu_pred.dtype)
            Sigma = eye.unsqueeze(0).expand(batch_size, -1, -1)

        # Optimal weights using predicted mu
        w_pred = self.opt_layer(mu_pred, Sigma)

        # Optimal weights using true mu (detached — this is the target)
        with torch.no_grad():
            w_opt = self.opt_layer(mu_true, Sigma)

        # Regret: (mu_true @ w_opt).mean() - (mu_true @ w_pred).mean()
        portfolio_opt = torch.sum(mu_true * w_opt, dim=1)
        portfolio_pred = torch.sum(mu_true * w_pred, dim=1)
        regret = portfolio_opt - portfolio_pred

        # Penalize excessive weight changes
        turnover_penalty = torch.mean(torch.sum(torch.abs(w_pred - w_opt), dim=1))

        return regret.mean() + 0.01 * turnover_penalty


class SPOModel(nn.Module):
    """End-to-end SPO model.

    Features -> MLP/LSTM -> mu_pred -> DiffOpt Layer -> weights -> SPO loss

    Architecture:
        Feature encoder (MLP or LSTM)
        -> mu prediction head
        -> optional covariance prediction head
        -> Differentiable optimization layer
        -> Decision loss (SPO)
    """

    def __init__(self, n_features: int, n_assets: int = 10, hidden_dim: int = 128,
                 use_lstm: bool = False, seq_len: int = 60,
                 predict_cov: bool = False, lambda_reg: float = 1.0, max_weight: float = 0.2):
        super().__init__()
        self.n_assets = n_assets
        self.predict_cov = predict_cov
        self.use_lstm = use_lstm

        if use_lstm:
            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.mu_head = nn.Linear(hidden_dim, n_assets)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu_head = nn.Linear(hidden_dim, n_assets)

        if predict_cov:
            self.cov_net = CovarianceNetwork(n_features if not use_lstm else hidden_dim, n_assets)

        self.opt_layer = DifferentiableMarkowitzLayer(n_assets, lambda_reg, max_weight)

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (mu_pred, Sigma, weights)."""
        batch_size = x.shape[0]

        if self.use_lstm:
            out, (h_n, c_n) = self.encoder(x)
            features = out[:, -1, :]  # (batch, hidden_dim)
        else:
            features = self.encoder(x)

        mu_pred = self.mu_head(features)

        if self.predict_cov:
            Sigma = self.cov_net(x.view(batch_size, -1))
        else:
            # Use identity covariance
            Sigma = torch.eye(self.n_assets, device=x.device, dtype=x.dtype)
            Sigma = Sigma.unsqueeze(0).expand(batch_size, -1, -1)

        weights = self.opt_layer(mu_pred, Sigma)
        return mu_pred, Sigma, weights

    def compute_spo_loss(self, mu_pred: torch.Tensor, mu_true: torch.Tensor,
                          Sigma: torch.Tensor = None) -> torch.Tensor:
        """Compute SPO regret loss."""
        return SPOLoss(self.n_assets)(mu_pred, mu_true, Sigma)
