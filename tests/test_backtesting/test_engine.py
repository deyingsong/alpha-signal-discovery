import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine import BacktestEngine

class TestBacktestEngine:
    
    def setup_method(self):
        self.engine = BacktestEngine(transaction_cost=0.005)
    
    def test_generate_signals(self):
        """Test signal generation from predictions."""
        predictions = np.array([0.01, -0.005, 0.002, -0.015])
        threshold = 0.005
        
        expected_signals = np.array([1, 0, 0, -1])
        actual_signals = self.engine.generate_signals(predictions, threshold)
        
        np.testing.assert_array_equal(actual_signals, expected_signals)
    
    def test_simulate_trading(self):
        """Test complete trading simulation."""
        predictions = np.array([0.01, -0.01, 0.005, -0.005])
        returns = np.array([0.008, -0.012, 0.003, -0.002])
        threshold = 0.005
        
        results = self.engine.simulate_trading(predictions, returns, threshold)
        
        # Check that results DataFrame has correct columns
        expected_columns = [
            'predictions', 'returns', 'signals', 'positions', 
            'position_changes', 'transaction_costs', 
            'gross_returns', 'net_returns', 
            'cumulative_gross', 'cumulative_net'
        ]
        
        assert list(results.columns) == expected_columns
        assert len(results) == len(predictions)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # One year of daily returns
        
        metrics = self.engine.calculate_performance_metrics(returns)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'annualized_volatility',
            'sharpe_ratio', 'max_drawdown', 'calmar_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))