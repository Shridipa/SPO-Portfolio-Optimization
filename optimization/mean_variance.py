"""Mean-variance portfolio optimization using cvxpy."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import N_ASSETS, MAX_WEIGHT


def shrinkage_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Shrinkage covariance via Ledoit-Wolf."""
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_


class MeanVarianceOptimizer:
    """Mean-variance portfolio optimizer with cvxpy.

    Solves: max  mu @ w - lambda * w @ Sigma @ w
    s.t.    sum(w)=1, w >= 0, w <= max_weight
    """

    def __init__(self, n_assets=N_ASSETS, lambda_reg=1.0, max_weight=MAX_WEIGHT, transaction_cost=0.0):
        self.n_assets = n_assets
        self.lambda_reg = lambda_reg
        self.max_weight = max_weight
        self.transaction_cost = transaction_cost
        self.prev_weights = None

    def optimize(self, mu: np.ndarray, Sigma: np.ndarray, prev_weights: np.ndarray = None) -> np.ndarray:
        """Return optimal portfolio weights.

        Args:
            mu: (n_assets,) expected returns
            Sigma: (n_assets, n_assets) covariance matrix
            prev_weights: (n_assets,) weights from previous period (for transaction costs)
        """
        w = cp.Variable(self.n_assets)
        # Objective: risk-adjusted return
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        objective = cp.Maximize(ret - self.lambda_reg * risk)

        # Transaction cost penalty
        if self.transaction_cost > 0 and prev_weights is not None:
            tc = self.transaction_cost * cp.norm1(w - prev_weights)
            objective = cp.Maximize(ret - self.lambda_reg * risk - tc)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= self.max_weight,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if w.value is None:
            # Fallback: equal weight
            return np.ones(self.n_assets) / self.n_assets

        weights = np.array(w.value).flatten()
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / weights.sum()  # renormalize
        self.prev_weights = weights
        return weights

    def efficient_frontier(self, mu: np.ndarray, Sigma: np.ndarray, n_points=20):
        """Compute efficient frontier points: (risk, return) pairs."""
        risks, returns = [], []
        lambdas = np.logspace(-2, 2, n_points)
        for lam in lambdas:
            w = cp.Variable(self.n_assets)
            ret = mu @ w
            risk = cp.quad_form(w, Sigma)
            prob = cp.Problem(cp.Maximize(ret - lam * risk),
                              [cp.sum(w) == 1, w >= 0, w <= self.max_weight])
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if w.value is not None:
                risks.append(np.sqrt(risk.value))
                returns.append(ret.value)
        return np.array(risks), np.array(returns)
