# Robust SPO Portfolio Optimization

An end-to-end framework integrating Machine Learning, Portfolio Optimization, and Differentiable Programming. This project builds a "Smart Predict then Optimize" (SPO) architecture, where the portfolio optimization engine acts as a differentiable layer within a neural network, allowing the model to minimize decision regret rather than simple prediction error (MSE).

## Features

- **End-to-End Differentiable Optimization**: Uses `cvxpylayers` to embed Markowitz Mean-Variance optimization into PyTorch models.
- **Smart Predict then Optimize (SPO)**: Directly minimizes portfolio regret instead of Mean Squared Error.
- **Advanced Deep Learning Models**: Supports MLP, LSTM, and Time Series Transformers for feature extraction and return prediction.
- **Distributionally Robust Optimization (DRO)**: Integrates ellipsoidal/Wasserstein uncertainty sets into the differentiable solver for robust, conservative weights.
- **Regime-Aware Asset Allocation**: Gaussian HMM for market regime detection (Bull, Bear, Sideways, High Volatility).
- **Realistic Constraints**: Includes $L_1$-norm transaction cost penalties, max-weight bounds, and Ledoit-Wolf shrinkage covariance estimation.
- **Comprehensive Backtesting Engine**: Walk-forward backtesting system with continuous rebalancing and detailed strategy metrics (Sharpe, Max Drawdown, Turnover).

## Repository Structure

```text
robust-spo-portfolio/
├── data/                  # Data ingestion and feature pipeline
├── features/              # Technical and Macro indicators
├── models/                # ML Models (Linear, XGBoost, LSTM, Transformer)
├── optimization/          # Classical MVO, CVaR, DRO, and DiffOpt Layer
├── losses/                # SPO regret loss functions
├── backtest/              # Walk-forward backtesting engine & metrics
├── regime/                # Hidden Markov Model market regime detection
├── experiments/           # Interactive Jupyter Notebooks
└── run_all.py             # Full pipeline orchestrator
```

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shridipa/SPO-Portfolio-Optimization.git
   cd SPO-Portfolio-Optimization
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn torch cvxpy cvxpylayers yfinance xgboost hmmlearn
   ```

### Running the Pipeline

You can run the entire evaluation pipeline (Baseline models, SPO, SPO+DRO, Regime Detection, Visualization) using the orchestrator:

```bash
# Run the full pipeline (this will take a while to train all neural nets)
python run_all.py

# Run a quick test with fewer epochs
python run_all.py --quick
```

Results and comparison tables will be saved to the `results/` directory.

### Interactive Experiments

Check out the `experiments/` directory for interactive Jupyter notebooks:
- `baseline.ipynb`: Compares Classical MVO to two-stage predict-then-optimize models.
- `spo.ipynb`: Trains and evaluates the differentiable SPO architecture.
- `spo_dro.ipynb`: Integrates SPO with Distributionally Robust Optimization.

## Research Context

This project extends classical approaches to portfolio management by framing it as a differentiable programming problem.

Traditional "Predict-then-Optimize" models suffer because a low prediction error (MSE) does not guarantee a high Sharpe ratio. By leveraging **SPO**, the neural network learns to predict specifically the features that maximize the portfolio's objective function. 

Combined with **DRO**, the model is taught to construct portfolios that are not only optimal for the predicted returns but also robust to the inherent noise and uncertainty of financial markets.
