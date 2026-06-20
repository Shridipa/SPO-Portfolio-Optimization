"""Linear regression model for return prediction."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TICKERS


class LinearReturnModel:
    """Linear regression predicting next-period returns for all assets."""

    def __init__(self):
        self.model = None
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit model.

        X: (n_samples, n_features) or (n_samples, n_assets * n_features)
        y: (n_samples, n_assets)  forward returns
        """
        self.feature_cols = list(X.columns)
        # Flatten multi-index if needed
        if isinstance(X.columns, pd.MultiIndex):
            X = X.copy()
            X.columns = [f"{a}_{b}" for a, b in X.columns]
            self.feature_cols = list(X.columns)

        base = LinearRegression()
        self.model = MultiOutputRegressor(base, n_jobs=-1)
        self.model.fit(X.values, y.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict next-period returns. Returns array (n_samples, n_assets)."""
        if isinstance(X.columns, pd.MultiIndex):
            X = X.copy()
            X.columns = [f"{a}_{b}" for a, b in X.columns]
        return self.model.predict(X.values)

    def get_feature_importance(self):
        """Return feature importance coefficients per asset."""
        if not hasattr(self.model, 'estimators_'):
            return None
        coefs = np.array([est.coef_ for est in self.model.estimators_])
        return pd.DataFrame(coefs.T, index=self.feature_cols, columns=TICKERS)
