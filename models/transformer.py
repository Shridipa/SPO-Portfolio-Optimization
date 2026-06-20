"""Time Series Transformer for return prediction."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class PositionalEncoding(nn.Module):
    """Positional encoding for time-series."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerPredictor(nn.Module):
    """Transformer for multi-asset return prediction.
    
    Input:  (batch, seq_len, n_features)
    Output: (batch, n_assets)
    """
    def __init__(self, n_features, n_assets=10, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, n_assets)
        
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_linear(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        # Pass through transformer encoder
        out = self.transformer_encoder(x)
        # Take the representation of the last time step
        last_hidden = out[:, -1, :] # (batch, d_model)
        return self.fc_out(last_hidden) # (batch, n_assets)

class TransformerReturnModel:
    """Wrapper for training and inference with TransformerPredictor."""

    def __init__(self, n_features, n_assets=10, seq_len=60,
                 d_model=128, nhead=4, num_layers=2, lr=1e-3, device=None):
        self.seq_len = seq_len
        self.n_assets = n_assets
        self.n_features = n_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TransformerPredictor(
            n_features, n_assets, d_model, nhead, num_layers
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _build_sequences(self, X: np.ndarray, y: np.ndarray):
        xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            xs.append(X[i - self.seq_len:i])
            ys.append(y[i])
        return np.array(xs), np.array(ys)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, epochs=100, batch_size=64, verbose=True):
        X_arr = X.values if not isinstance(X.columns, pd.MultiIndex) else X.values
        y_arr = y.values

        X_seq, y_seq = self._build_sequences(X_arr, y_arr)
        if len(X_seq) == 0:
            return

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        X_arr = X.values if not isinstance(X.columns, pd.MultiIndex) else X.values
        X_seq, _ = self._build_sequences(X_arr, np.zeros((len(X_arr), self.n_assets)))
        
        if len(X_seq) == 0:
            return np.full((len(X_arr), self.n_assets), np.nan)
            
        with torch.no_grad():
            preds = self.model(torch.tensor(X_seq, dtype=torch.float32).to(self.device))
            
        full = np.full((len(X_arr), self.n_assets), np.nan)
        full[self.seq_len:] = preds.cpu().numpy()
        return full
