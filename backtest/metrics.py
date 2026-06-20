"""Portfolio performance metrics."""
import numpy as np
import pandas as pd


def compute_metrics(returns: np.ndarray, weights_history: np.ndarray = None,
                    benchmark_returns: np.ndarray = None, risk_free_rate: float = 0.0,
                    freq: int = 252) -> dict:
    """Compute comprehensive portfolio performance metrics.

    Args:
        returns: (T,) portfolio returns
        weights_history: (T, n_assets) portfolio weights over time (for turnover)
        benchmark_returns: (T,) benchmark returns (e.g. SPY)
        risk_free_rate: annualized risk-free rate
        freq: number of periods per year (252 daily, 12 monthly)

    Returns:
        dict of metrics
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)

    # Cumulative returns
    cum_ret = np.cumprod(1 + returns) - 1
    total_return = cum_ret[-1]

    # Annualized return
    ann_return = (1 + total_return) ** (freq / T) - 1

    # Annualized volatility
    ann_vol = np.std(returns, ddof=1) * np.sqrt(freq)

    # Sharpe ratio
    excess_returns = returns - risk_free_rate / freq
    sharpe = np.sqrt(freq) * np.mean(excess_returns) / np.std(returns, ddof=1) if np.std(returns) > 1e-10 else 0.0

    # Sortino ratio
    downside = returns[returns < 0]
    downside_vol = np.std(downside, ddof=1) * np.sqrt(freq) if len(downside) > 1 else 1e-10
    sortino = np.sqrt(freq) * np.mean(excess_returns) / downside_vol if downside_vol > 1e-10 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(np.cumprod(1 + returns))
    drawdown = (np.cumprod(1 + returns) - peak) / peak
    max_drawdown = np.min(drawdown)

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win rate
    win_rate = np.sum(returns > 0) / T

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')

    # Turnover
    turnover = 0.0
    if weights_history is not None and len(weights_history) > 1:
        turnover = np.mean(np.sum(np.abs(np.diff(weights_history, axis=0)), axis=1)) * freq

    # Alpha and Beta vs benchmark
    alpha, beta = 0.0, 0.0
    if benchmark_returns is not None:
        benchmark = np.asarray(benchmark_returns, dtype=float)
        min_len = min(len(returns), len(benchmark))
        r = returns[:min_len]
        b = benchmark[:min_len]
        cov = np.cov(r, b)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else 0.0
        alpha = (np.mean(r) - beta * np.mean(b)) * freq

    # VaR and CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if np.sum(returns <= var_95) > 0 else var_95

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "turnover": turnover,
        "alpha": alpha,
        "beta": beta,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def metrics_to_dataframe(metrics_dict: dict, name: str = "") -> pd.DataFrame:
    """Convert nested {model_name: {metric: value}} dict to a clean DataFrame."""
    rows = []
    for model_name, metrics in metrics_dict.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index("model")
    return df.round(4)
