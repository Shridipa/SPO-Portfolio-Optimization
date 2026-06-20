"""Walk-forward backtesting engine for portfolio strategies."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimization.mean_variance import MeanVarianceOptimizer, shrinkage_covariance


class BacktestEngine:
    """Walk-forward backtesting engine.

    At each rebalance date:
        1. Predict returns using latest features (if model + features provided)
        2. Estimate covariance matrix from historical returns
        3. Run portfolio optimizer → get weights
        4. Compute forward portfolio return until next rebalance
    """

    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame,
                 rebalance_freq: str = "monthly",
                 train_window: int = 252 * 5,
                 lambda_reg: float = 1.0,
                 max_weight: float = 0.2,
                 transaction_cost: float = 0.0):
        self.prices = prices
        self.returns = returns
        self.rebalance_freq = rebalance_freq
        self.train_window = train_window
        self.lambda_reg = lambda_reg
        self.max_weight = max_weight
        self.transaction_cost = transaction_cost
        self.n_assets = prices.shape[1]

        self.rebalance_dates = self._get_rebalance_dates()

        self.weights_history = []
        self.weights_dates = []
        self.portfolio_returns = []

    def _get_rebalance_dates(self):
        idx = self.prices.index
        if self.rebalance_freq == "monthly":
            rebalance = idx[::21]
        elif self.rebalance_freq == "weekly":
            rebalance = idx[::5]
        else:
            rebalance = idx
        train_cutoff = idx[self.train_window] if self.train_window < len(idx) else idx[0]
        rebalance = rebalance[rebalance > train_cutoff]
        return pd.DatetimeIndex(rebalance)

    def run(self, mu_predictor=None, use_shrinkage=True, verbose=True):
        """Run walk-forward backtest.

        Args:
            mu_predictor: callable(date) -> (n_assets,) mu vector.
                          If None, uses historical mean of training returns.
            use_shrinkage: use Ledoit-Wolf shrinkage covariance
        """
        n = len(self.rebalance_dates)
        if verbose:
            print(f"Running backtest over {n} rebalance periods...")

        for idx, rebal_date in enumerate(self.rebalance_dates):
            train_start = rebal_date - pd.Timedelta(days=self.train_window)
            train_returns = self.returns.loc[train_start:rebal_date].dropna()
            if len(train_returns) < 20:
                continue

            # Predict expected returns
            if mu_predictor is not None:
                mu = mu_predictor(rebal_date)
                if isinstance(mu, np.ndarray) and mu.ndim == 1 and len(mu) == self.n_assets:
                    pass  # good
                else:
                    mu = train_returns.mean().values
            else:
                mu = train_returns.mean().values

            # Estimate covariance
            if use_shrinkage:
                Sigma = shrinkage_covariance(train_returns)
            else:
                Sigma = train_returns.cov().values * 252

            # Optimize
            prev_w = self.weights_history[-1] if self.weights_history else None
            opt = MeanVarianceOptimizer(
                n_assets=self.n_assets,
                lambda_reg=self.lambda_reg,
                max_weight=self.max_weight,
                transaction_cost=self.transaction_cost,
            )
            w = opt.optimize(mu, Sigma, prev_weights=prev_w)

            self.weights_history.append(w)
            self.weights_dates.append(rebal_date)

            # Portfolio return to next rebalance
            if idx + 1 < n:
                next_date = self.rebalance_dates[idx + 1]
            else:
                next_date = self.prices.index[-1]

            period_returns = self.returns.loc[rebal_date:next_date].values
            if len(period_returns) > 0:
                port_ret = (period_returns * w).sum(axis=1).mean()
                self.portfolio_returns.append(port_ret)

            if verbose and (idx + 1) % 20 == 0:
                print(f"  Period {idx+1}/{n}")

        self.weights_history = np.array(self.weights_history)
        self.portfolio_returns = np.array(self.portfolio_returns)

        if verbose:
            non_nan = self.portfolio_returns[~np.isnan(self.portfolio_returns)]
            print(f"Backtest complete. {len(self.portfolio_returns)} periods.")
            print(f"Mean portfolio return: {np.nanmean(self.portfolio_returns):.6f}")
            print(f"Std portfolio return:  {np.nanstd(self.portfolio_returns):.6f}")
            if len(non_nan) > 1:
                print(f"Sharpe (est):         {np.sqrt(12) * np.mean(non_nan) / np.std(non_nan):.4f}")

        return self

    def get_results(self):
        return {
            "returns": self.portfolio_returns,
            "weights": self.weights_history,
            "weights_dates": self.weights_dates,
            "rebalance_dates": self.rebalance_dates,
        }
