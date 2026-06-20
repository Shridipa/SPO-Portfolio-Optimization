"""CVaR (Conditional Value at Risk) portfolio optimization using cvxpy."""
import sys
from pathlib import Path
import numpy as np
import cvxpy as cp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import N_ASSETS, MAX_WEIGHT


class CVaROptimizer:
    """CVaR portfolio optimizer.

    Minimizes CVaR at level alpha (e.g. 0.05) given a return scenarios matrix.
    """

    def __init__(self, n_assets=N_ASSETS, alpha=0.05, max_weight=MAX_WEIGHT):
        self.n_assets = n_assets
        self.alpha = alpha
        self.max_weight = max_weight

    def optimize(self, scenarios: np.ndarray, target_return: float = None) -> np.ndarray:
        """Return weights minimizing CVaR.

        Args:
            scenarios: (n_scenarios, n_assets) sampled return scenarios
            target_return: if set, constrain expected return >= target_return
        """
        n_scenarios = scenarios.shape[0]
        w = cp.Variable(self.n_assets)
        z = cp.Variable(n_scenarios)  # tail losses
        VaR = cp.Variable()  # Value at Risk

        # Portfolio scenarios
        port_scenarios = scenarios @ w

        # CVaR objective: VaR + 1/(alpha * S) * sum(z)
        objective = cp.Minimize(VaR + 1.0 / (self.alpha * n_scenarios) * cp.sum(z))

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0,
            w <= self.max_weight,
            z >= 0,
            port_scenarios >= VaR - z,  # z >= VaR - portfolio_return
        ]

        if target_return is not None:
            constraints.append(cp.mean(port_scenarios) >= target_return)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if w.value is None:
            return np.ones(self.n_assets) / self.n_assets

        weights = np.array(w.value).flatten()
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / weights.sum()
        return weights
