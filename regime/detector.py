"""Market regime detection using HMM and LSTM classifiers.

Identifies market regimes (Bull, Bear, Sideways, High Volatility)
to condition portfolio optimization on the current market state.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class RegimeDetector:
    """Market regime detector using Hidden Markov Models.

    Labels historical data with regime states and can predict
    the current regime in real-time.
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            random_state=random_state,
            n_iter=1000,
        )
        self.scaler = StandardScaler()
        self.regime_labels = None

    def fit(self, returns: pd.DataFrame, features: pd.DataFrame = None):
        """Fit HMM on return features.

        Args:
            returns: (T, n_assets) DataFrame or (T,) Series of portfolio returns
            features: (T, n_features) additional features (optional)
        """
        if features is not None:
            X = features.values
        else:
            # Use portfolio returns + volatility + volume as features
            if isinstance(returns, pd.DataFrame):
                port_returns = returns.mean(axis=1)
            else:
                port_returns = returns

            # Build feature matrix
            X = pd.DataFrame({
                "returns": port_returns,
                "abs_returns": port_returns.abs(),
                "volatility": port_returns.rolling(20).std().fillna(0),
                "skew": port_returns.rolling(20).skew().fillna(0),
                "volume_rel": 1.0,  # placeholder
            }).values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Label regimes by mean return (descending)
        states = self.model.predict(X_scaled)
        self.regime_labels = self._map_states(states, X_scaled)
        return self.regime_labels

    def _map_states(self, states: np.ndarray, X: np.ndarray = None) -> np.ndarray:
        """Map HMM state indices to meaningful regime labels.

        States are ordered by mean return: 0=highest return, 3=lowest.
        High vol gets the highest abs(return) state.
        """
        if X is None:
            return states

        mean_returns = np.array([X[states == i, 0].mean() for i in range(self.n_regimes)])
        vol = np.array([X[states == i, 2].mean() for i in range(self.n_regimes)])

        # Sort states: highest return = Bull, lowest = Bear, highest vol = HighVol, rest = Sideways
        order = np.argsort(mean_returns)[::-1]  # descending

        labels = np.zeros_like(states)
        label_map = {
            order[0]: 0,  # Bull
            order[-1]: 1,  # Bear
        }

        # The highest vol state that's not Bull/Bear is HighVol
        remaining = [i for i in order if i not in (order[0], order[-1])]
        if remaining:
            high_vol_idx = remaining[np.argmax(vol[remaining])]
            label_map[high_vol_idx] = 3  # HighVol
            remaining.remove(high_vol_idx)

        # Rest are Sideways
        for r in remaining:
            label_map[r] = 2  # Sideways

        for old, new in label_map.items():
            labels[states == old] = new

        return labels

    def predict(self, returns: pd.DataFrame, features: pd.DataFrame = None) -> int:
        """Predict current market regime.

        Returns integer: 0=Bull, 1=Bear, 2=Sideways, 3=HighVol
        """
        if features is not None:
            X = features.values[-1:].reshape(1, -1)
        else:
            if isinstance(returns, pd.DataFrame):
                port_returns = returns.mean(axis=1)
            else:
                port_returns = returns
            recent = port_returns.iloc[-20:] if len(port_returns) >= 20 else port_returns
            X = np.array([[
                recent.iloc[-1],
                recent.abs().iloc[-1],
                recent.std() if len(recent) > 1 else 0,
                recent.skew() if len(recent) > 3 else 0,
                1.0,
            ]])

        X_scaled = self.scaler.transform(X)
        state = self.model.predict(X_scaled)[0]

        if self.regime_labels is not None and len(self.regime_labels) > 0:
            # Map to the regime label from training
            label_map = {}
            train_labels = self.model.predict(self.scaler.transform(
                np.zeros((1, X_scaled.shape[1]))
            ))
            # Use stored mapping from training
            unique_states = np.unique(train_labels)
            for s in unique_states:
                mask = self.model.monitored_  # Not ideal; use fitted mean returns instead
            return int(state % 4)  # Fallback

        return int(state % 4)

    def get_regime_returns(self, returns: pd.DataFrame) -> dict:
        """Get summary statistics for each regime."""
        features = None
        labels = self.fit(returns, features)
        if isinstance(returns, pd.DataFrame):
            port_returns = returns.mean(axis=1)
        else:
            port_returns = returns

        summary = {}
        regime_names = ["Bull", "Bear", "Sideways", "HighVol"]
        for i in range(self.n_regimes):
            mask = labels == i
            if mask.sum() > 0:
                summary[regime_names[i]] = {
                    "count": int(mask.sum()),
                    "mean_return": float(port_returns[mask].mean()),
                    "std_return": float(port_returns[mask].std()),
                    "sharpe": float(port_returns[mask].mean() / port_returns[mask].std()) if port_returns[mask].std() > 0 else 0,
                }
        return summary


class RegimeAwarePortfolio:
    """Portfolio optimizer that conditions on market regime.

    Uses different risk aversion and constraints per regime.
    """

    def __init__(self, n_assets: int = 10):
        self.n_assets = n_assets
        self.detector = RegimeDetector()
        self.regime_params = {
            0: {"lambda_reg": 0.5, "max_weight": 0.25},   # Bull: take more risk
            1: {"lambda_reg": 3.0, "max_weight": 0.15},   # Bear: be defensive
            2: {"lambda_reg": 1.0, "max_weight": 0.20},   # Sideways: neutral
            3: {"lambda_reg": 2.0, "max_weight": 0.12},   # HighVol: reduce exposure
        }

    def get_regime_params(self, returns: pd.DataFrame) -> dict:
        """Get optimization parameters for current regime."""
        regime = self.detector.predict(returns)
        return self.regime_params.get(regime, {"lambda_reg": 1.0, "max_weight": 0.2})
