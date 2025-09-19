def calculate_advanced_metrics(returns):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
    
    # Advanced metrics
    metrics['sortino_ratio'] = returns.mean() / returns[returns < 0].std() * np.sqrt(252)
    metrics['calmar_ratio'] = metrics['total_return'] / max_drawdown(returns)
    metrics['omega_ratio'] = omega_ratio(returns)
    metrics['tail_ratio'] = returns.quantile(0.95) / abs(returns.quantile(0.05))
    
    # Rolling performance
    metrics['rolling_sharpe_12m'] = returns.rolling(252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252)
    )
    
    return metrics