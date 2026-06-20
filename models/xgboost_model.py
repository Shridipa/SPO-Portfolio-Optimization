"""XGBoost model for return prediction."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class XGBoostReturnModel:
    """XGBoost regressor for multi-asset return prediction."""

    def __init__(self, n_estimators=500, max_depth=5, learning_rate=0.01, early_stopping_rounds=None):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": 42,
        }
        self.models = {}
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_set=None):
        self.feature_cols = list(X.columns)
        if isinstance(X.columns, pd.MultiIndex):
            X = X.copy()
            X.columns = [f"{a}_{b}" for a, b in X.columns]
            self.feature_cols = list(X.columns)

        for i, col in enumerate(y.columns):
            model = xgb.XGBRegressor(**self.params)
            evals = None
            if eval_set is not None:
                evals = [(eval_set[0].values, eval_set[1].values[:, i])]
            model.fit(
                X.values, y.values[:, i],
                eval_set=evals,
                verbose=False
            )
            self.models[col] = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X.columns, pd.MultiIndex):
            X = X.copy()
            X.columns = [f"{a}_{b}" for a, b in X.columns]
        cols = sorted(self.models.keys())
        preds = np.column_stack([self.models[c].predict(X.values) for c in cols])
        return preds

    def get_feature_importance(self):
        if not self.models:
            return None
        first_model = list(self.models.values())[0]
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": first_model.feature_importances_
        }).sort_values("importance", ascending=False)
