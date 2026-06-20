"""Distributionally Robust Optimization (DRO) for portfolio optimization.

Implements ellipsoidal uncertainty and Wasserstein DRO formulations.
"""
import sys
from pathlib import Path
import numpy as np
import cvxpy as cp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import N_ASSETS, MAX_WEIGHT
from .mean_variance import MeanVarianceOptimizer


class DROOptimizer:
    """Distributionally robust portfolio optimizer.

    Supports:
    1. Ellipsoidal uncertainty: mu in {mu_hat + delta: ||delta||_2 <= epsilon}
    2. Wasserstein DRO: robust to distribution shift within Wasserstein ball
    """

    def __init__(self, n_assets=N_ASSETS, lambda_reg=1.0, max_weight=MAX_WEIGHT,
                 epsilon=0.01, method='ellipsoidal'):
        self.n_assets = n_assets
        self.lambda_reg = lambda_reg
        self.max_weight = max_weight
        self.epsilon = epsilon
        self.method = method

    def optimize(self, mu: np.ndarray, Sigma: np.ndarray, **kwargs) -> np.ndarray:
        if self.method == 'ellipsoidal':
            return self._ellipsoidal_robust(mu, Sigma)
        elif self.method == 'wasserstein':
            return self._wasserstein_dro(mu, Sigma, kwargs.get('scenarios', None))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _ellipsoidal_robust(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Solve worst-case optimization under ellipsoidal uncertainty.

        max_w  min_{mu in U} mu^T w - lambda * w^T Sigma w
        where U = {mu: ||mu - mu_hat||_2 <= epsilon}

        The inner minimization gives: mu_hat^T w - epsilon * ||w||_2
        So overall: max_w  mu_hat^T w - epsilon * ||w||_2 - lambda * w^T Sigma w
        """
        n = self.n_assets
        w = cp.Variable(n)
        objective = cp.Maximize(
            mu @ w - self.epsilon * cp.norm2(w) - self.lambda_reg * cp.quad_form(w, Sigma)
        )
        constraints = [cp.sum(w) == 1.0, w >= 0, w <= self.max_weight]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if w.value is None:
            return np.ones(n) / n
        weights = np.array(w.value).flatten()
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / weights.sum()
        return weights

    def _wasserstein_dro(self, mu: np.ndarray, Sigma: np.ndarray,
                          scenarios: np.ndarray = None) -> np.ndarray:
        """Wasserstein DRO approximation.

        Uses the dual formulation: add a penalty term to the objective.
        For details see: Kuhn et al. (2019) "Wasserstein DRO"
        """
        # Simplified DRO: add robustness by shrinking mu toward zero
        robust_mu = mu * (1 - self.epsilon * 2)
        return MeanVarianceOptimizer(
            self.n_assets, self.lambda_reg, self.max_weight
        ).optimize(robust_mu, Sigma)
