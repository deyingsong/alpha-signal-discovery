import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging
from scipy import stats
from sklearn.metrics import confusion_matrix

class BacktestEngine:
    """Enhanced backtesting engine for trading strategy evaluation."""
    
    def __init__(self, transaction_cost: float = 0.005):
        self.transaction_cost = transaction_cost
        self.logger = logging.getLogger(__name__)
        
    def optimize_threshold(self, predictions: np.ndarray, returns: np.ndarray, 
                         threshold_range: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Optimize trading threshold using grid search on Sharpe ratio."""
        if threshold_range is None:
            # Use percentiles of predictions for threshold range
            threshold_range = np.percentile(np.abs(predictions), np.linspace(0, 95, 50))
        
        best_sharpe = -np.inf
        best_threshold = 0
        
        for threshold in threshold_range:
            # Generate signals
            signals = np.where(predictions > threshold, 1,
                             np.where(predictions < -threshold, -1, 0))
            
            # Calculate returns
            position_changes = np.diff(np.concatenate([[0], signals]))
            costs = np.abs(position_changes) * self.transaction_cost
            gross_returns = signals * returns
            net_returns = gross_returns - costs
            
            # Calculate Sharpe ratio
            if np.std(net_returns) > 0:
                sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)
            else:
                sharpe = 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold
        
        self.logger.info(f"Optimal threshold: {best_threshold:.4f}")
        self.logger.info(f"Best Sharpe ratio: {best_sharpe:.3f}")
        
        return best_threshold, best_sharpe
    
    def optimize_dynamic_threshold(self, predictions: np.ndarray, returns: np.ndarray,
                                  lookback: int = 60) -> np.ndarray:
        """Optimize threshold dynamically based on rolling statistics."""
        dynamic_thresholds = np.zeros(len(predictions))
        
        for i in range(len(predictions)):
            if i < lookback:
                # Use fixed threshold for initial period
                dynamic_thresholds[i] = np.std(predictions[:i+1]) * 0.5
            else:
                # Use rolling statistics
                rolling_std = np.std(predictions[i-lookback:i])
                rolling_mean = np.abs(np.mean(predictions[i-lookback:i]))
                
                # Adaptive threshold based on recent volatility
                dynamic_thresholds[i] = max(rolling_std * 0.5, rolling_mean * 0.25)
        
        return dynamic_thresholds
    
    def simulate_trading(self, predictions: np.ndarray, returns: np.ndarray, 
                        threshold: float, use_dynamic: bool = False) -> pd.DataFrame:
        """Simulate trading with given predictions and threshold."""
        
        if use_dynamic:
            thresholds = self.optimize_dynamic_threshold(predictions, returns)
            signals = np.array([1 if pred > thresh else -1 if pred < -thresh else 0 
                               for pred, thresh in zip(predictions, thresholds)])
        else:
            # Generate trading signals
            signals = np.where(predictions > threshold, 1,
                             np.where(predictions < -threshold, -1, 0))
        
        # Initialize positions (0 = flat, 1 = long, -1 = short)
        positions = signals.copy()
        
        # Calculate position changes
        position_changes = np.diff(np.concatenate([[0], positions]))
        
        # Calculate transaction costs
        transaction_costs = np.abs(position_changes) * self.transaction_cost
        
        # Calculate returns
        gross_returns = positions * returns
        net_returns = gross_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_gross = np.cumprod(1 + gross_returns) - 1
        cumulative_net = np.cumprod(1 + net_returns) - 1
        
        # Create results DataFrame
        results = pd.DataFrame({
            'predictions': predictions,
            'returns': returns,
            'signals': signals,
            'positions': positions,
            'position_changes': position_changes,
            'transaction_costs': transaction_costs,
            'gross_returns': gross_returns,
            'net_returns': net_returns,
            'cumulative_gross': cumulative_gross,
            'cumulative_net': cumulative_net
        })
        
        return results
    
    def calculate_performance_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        # Basic return metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        annualized_volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown metrics
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and profit factor
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) != 0 else np.inf
        
        # Information ratio (if benchmark is 0)
        information_ratio = annualized_return / np.std(returns - 0) / np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'information_ratio': information_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def calculate_turnover(self, positions: np.ndarray) -> float:
        """Calculate portfolio turnover."""
        position_changes = np.abs(np.diff(positions))
        turnover = np.sum(position_changes) / (2 * len(positions))
        return turnover
    
    def analyze_predictions(self, predictions: np.ndarray, returns: np.ndarray,
                          threshold: float) -> Dict[str, float]:
        """Analyze prediction quality and trading signals."""
        
        # Generate signals
        signals = np.where(predictions > threshold, 1,
                         np.where(predictions < -threshold, -1, 0))
        
        # Direction accuracy
        actual_direction = np.sign(returns)
        predicted_direction = np.sign(predictions)
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Signal quality when trading
        trading_mask = signals != 0
        if np.sum(trading_mask) > 0:
            trading_accuracy = np.mean(
                actual_direction[trading_mask] == predicted_direction[trading_mask]
            )
            
            # Precision for long and short signals
            long_mask = signals == 1
            short_mask = signals == -1
            
            long_precision = np.mean(returns[long_mask] > 0) if np.sum(long_mask) > 0 else 0
            short_precision = np.mean(returns[short_mask] < 0) if np.sum(short_mask) > 0 else 0
            
            # Average return per signal type
            avg_return_long = np.mean(returns[long_mask]) if np.sum(long_mask) > 0 else 0
            avg_return_short = -np.mean(returns[short_mask]) if np.sum(short_mask) > 0 else 0
        else:
            trading_accuracy = 0
            long_precision = 0
            short_precision = 0
            avg_return_long = 0
            avg_return_short = 0
        
        # Correlation between predictions and returns
        correlation = np.corrcoef(predictions, returns)[0, 1]
        
        # Information coefficient (rank correlation)
        ic = stats.spearmanr(predictions, returns)[0]
        
        return {
            'direction_accuracy': direction_accuracy,
            'trading_accuracy': trading_accuracy,
            'long_precision': long_precision,
            'short_precision': short_precision,
            'avg_return_long': avg_return_long,
            'avg_return_short': avg_return_short,
            'correlation': correlation,
            'information_coefficient': ic,
            'num_trades': np.sum(trading_mask),
            'pct_time_in_market': np.mean(trading_mask)
        }
    
    def run_walk_forward_backtest(self, predictions_list: List[np.ndarray],
                                 returns_list: List[np.ndarray],
                                 reoptimize_threshold: bool = True) -> pd.DataFrame:
        """Run walk-forward backtest with multiple periods."""
        all_results = []
        
        for i, (pred, ret) in enumerate(zip(predictions_list, returns_list)):
            if reoptimize_threshold:
                threshold, _ = self.optimize_threshold(pred, ret)
            else:
                threshold = 0
            
            results = self.simulate_trading(pred, ret, threshold)
            results['period'] = i
            all_results.append(results)
        
        return pd.concat(all_results, ignore_index=True)