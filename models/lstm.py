"""LSTM model for return prediction."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class LSTMPredictor(nn.Module):
    """LSTM for multi-asset return prediction.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_assets)
    """

    def __init__(self, n_features, n_assets=10, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, n_assets)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        # Take last hidden state
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last_hidden)  # (batch, n_assets)


class LSTMReturnModel:
    """Wrapper for training and inference with LSTMPredictor."""

    def __init__(self, n_features, n_assets=10, seq_len=60,
                 hidden_size=128, num_layers=2, lr=1e-3, device=None):
        self.seq_len = seq_len
        self.n_assets = n_assets
        self.n_features = n_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMPredictor(n_features, n_assets, hidden_size, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _build_sequences(self, X: np.ndarray, y: np.ndarray):
        """X: (T, n_features), y: (T, n_assets). Returns sequences and targets."""
        xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            xs.append(X[i - self.seq_len:i])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_set=None, epochs=100, batch_size=64, verbose=True):
        X_arr = X.values if not isinstance(X.columns, pd.MultiIndex) else X.values
        y_arr = y.values

        X_seq, y_seq = self._build_sequences(X_arr, y_arr)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.loss_fn(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_arr = X.values
        X_seq, _ = self._build_sequences(X_arr, np.zeros((len(X_arr), self.n_assets)))
        with torch.no_grad():
            preds = self.model(torch.tensor(X_seq, dtype=torch.float32).to(self.device))
        # Pad first seq_len predictions with NaN
        full = np.full((len(X_arr), self.n_assets), np.nan)
        full[self.seq_len:] = preds.cpu().numpy()
        return full
