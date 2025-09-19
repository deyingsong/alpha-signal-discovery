import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Dict, Any, List
from sklearn.metrics import mean_squared_error
import logging

class TimeSeriesCrossValidator:
    """Walk-forward cross-validation for time series data."""

    def __init__(self, n_splits: int = 5, test_size: int = 90, gap: int = 1):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap  # Embargo period
        self.logger = logging.getLogger(__name__)
    
    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits for walk-forward validation."""
        n_samples = len(X)
        
        # Calculate split points
        initial_train_size = (n_samples - self.n_splits * self.test_size) // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Expanding window approach
            train_end = initial_train_size + i * self.test_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def cross_validate_model(self, model_class, model_config: Dict[str, Any], 
                           X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation and return performance metrics."""
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X)):
            # Split data
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Initialize and train model
            model = model_class(model_config)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            fold_scores.append({
                'fold': fold,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            self.logger.info(f"Fold {fold}: RMSE = {np.sqrt(mse):.6f}")
        
        # Aggregate results
        avg_mse = np.mean([score['mse'] for score in fold_scores])
        avg_rmse = np.mean([score['rmse'] for score in fold_scores])
        
        return {
            'avg_mse': avg_mse,
            'avg_rmse': avg_rmse,
            'fold_scores': fold_scores
        }