"""
MLP Model that integrates with your existing BaseModel architecture
Optimized for small financial datasets (1,500 samples)
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
from .base_model import BaseModel
import logging
from typing import Dict, Any, Optional
from scipy.stats import uniform, randint

class MLPModel(BaseModel):
    """
    MLP (Multi-Layer Perceptron) model optimized for small financial datasets.
    Uses sklearn's MLPRegressor with heavy regularization.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.use_fast_mode = config.get('fast_mode', True)
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit MLP with appropriate configuration for dataset size.
        """
        n_samples = len(X)
        n_features = X.shape[1]
        
        self.logger.info(f"Training MLP on {n_samples} samples with {n_features} features")
        
        if n_samples < 2000:
            self._fit_tiny_mlp(X, y)
        elif n_samples < 10000:
            self._fit_small_mlp(X, y)
        else:
            self._fit_standard_mlp(X, y)
        
        self.is_fitted = True
    
    def _fit_tiny_mlp(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Ultra-small MLP for <2000 samples.
        Uses fixed small architecture to avoid overfitting.
        """
        self.logger.info("Using ultra-tiny MLP configuration for small dataset")
        
        if self.use_fast_mode:
            # Fixed configuration that works well
            self.model = MLPRegressor(
                hidden_layer_sizes=(8,),  # Single tiny layer
                activation='relu',
                solver='lbfgs',  # Better for small datasets
                alpha=1.0,  # Heavy L2 regularization
                max_iter=1000,
                random_state=self.config.get('random_state', 42),
                early_stopping=False,  # lbfgs doesn't support early stopping
                learning_rate_init=0.01
            )
            
            self.model.fit(X, y)
            self.logger.info(f"Trained tiny MLP with {self._count_parameters()} parameters")
            
        else:
            # Minimal grid search
            param_grid = {
                'hidden_layer_sizes': [(8,), (16,), (8, 4)],
                'alpha': [0.1, 1.0, 10.0],
                'learning_rate_init': [0.001, 0.01]
            }
            
            base_mlp = MLPRegressor(
                activation='relu',
                solver='lbfgs',
                max_iter=1000,
                random_state=self.config.get('random_state', 42)
            )
            
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=2)  # Minimal CV
            
            self.model = GridSearchCV(
                base_mlp,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            self.model.fit(X, y)
            
            self.logger.info(f"Best params: {self.model.best_params_}")
            self.logger.info(f"Best CV score: {self.model.best_score_:.6f}")
    
    def _fit_small_mlp(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Small MLP for 2000-10000 samples.
        Can use slightly larger architecture.
        """
        self.logger.info("Using small MLP configuration")
        
        if self.use_fast_mode:
            # Good default for small-medium data
            self.model = MLPRegressor(
                hidden_layer_sizes=(32, 16),  # Two small layers
                activation='relu',
                solver='adam',
                alpha=0.1,  # Moderate regularization
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                shuffle=True,
                random_state=self.config.get('random_state', 42),
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                momentum=0.9
            )
            
            self.model.fit(X, y)
            
        else:
            # More comprehensive search
            param_distributions = {
                'hidden_layer_sizes': [(16,), (32,), (16, 8), (32, 16)],
                'alpha': uniform(0.01, 1.0),
                'learning_rate_init': uniform(0.0001, 0.01),
                'batch_size': ['auto', 32, 64]
            }
            
            base_mlp = MLPRegressor(
                activation='relu',
                solver='adam',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=self.config.get('random_state', 42)
            )
            
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            self.model = RandomizedSearchCV(
                base_mlp,
                param_distributions,
                n_iter=10,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.config.get('random_state', 42)
            )
            
            self.model.fit(X, y)
            
            self.logger.info(f"Best params: {self.model.best_params_}")
            self.logger.info(f"Best CV score: {self.model.best_score_:.6f}")
    
    def _fit_standard_mlp(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Standard MLP for >10000 samples.
        Can use deeper architecture with less regularization.
        """
        self.logger.info("Using standard MLP configuration for larger dataset")
        
        param_grid = {
            'hidden_layer_sizes': self.config.get('hidden_layer_sizes', 
                                                  [(50,), (100,), (50, 25), (100, 50)]),
            'activation': self.config.get('activation', ['relu', 'tanh']),
            'alpha': self.config.get('alpha_range', [0.0001, 0.001, 0.01]),
            'learning_rate': self.config.get('learning_rate', ['constant', 'adaptive']),
            'learning_rate_init': self.config.get('learning_rate_init', [0.001, 0.01])
        }
        
        base_mlp = MLPRegressor(
            solver='adam',
            max_iter=self.config.get('max_iter', 1000),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=self.config.get('n_iter_no_change', 30),
            random_state=self.config.get('random_state', 42)
        )
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.config.get('cv_folds', 3))
        
        self.model = GridSearchCV(
            base_mlp,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        
        self.logger.info(f"Best params: {self.model.best_params_}")
        self.logger.info(f"Best CV score: {self.model.best_score_:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def _count_parameters(self) -> int:
        """
        Estimate the number of parameters in the MLP.
        """
        if hasattr(self.model, 'coefs_'):
            # Count weights and biases
            n_params = 0
            for i in range(len(self.model.coefs_)):
                n_params += self.model.coefs_[i].size
                n_params += self.model.intercepts_[i].size
            return n_params
        elif hasattr(self.model, 'best_estimator_'):
            # GridSearchCV case
            n_params = 0
            for i in range(len(self.model.best_estimator_.coefs_)):
                n_params += self.model.best_estimator_.coefs_[i].size
                n_params += self.model.best_estimator_.intercepts_[i].size
            return n_params
        return 0
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if hasattr(self.model, 'best_params_'):
            return self.model.best_params_
        elif hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = {
            'model_type': 'MLP',
            'parameters': self._count_parameters(),
            'is_fitted': self.is_fitted
        }
        
        if hasattr(self.model, 'n_iter_'):
            info['iterations'] = self.model.n_iter_
        
        if hasattr(self.model, 'loss_'):
            info['final_loss'] = self.model.loss_
        
        if hasattr(self.model, 'best_params_'):
            info['best_params'] = self.model.best_params_
            
        return info
    
    def get_convergence_info(self) -> Optional[Dict[str, Any]]:
        """
        Get convergence information for diagnostics.
        """
        if hasattr(self.model, 'loss_curve_'):
            return {
                'loss_curve': self.model.loss_curve_,
                'n_iter': self.model.n_iter_,
                'converged': self.model.n_iter_ < self.model.max_iter
            }
        elif hasattr(self.model, 'best_estimator_'):
            if hasattr(self.model.best_estimator_, 'loss_curve_'):
                return {
                    'loss_curve': self.model.best_estimator_.loss_curve_,
                    'n_iter': self.model.best_estimator_.n_iter_,
                    'converged': self.model.best_estimator_.n_iter_ < self.model.best_estimator_.max_iter
                }
        return None
    
    @staticmethod
    def get_recommended_config(n_samples: int) -> Dict[str, Any]:
        """
        Get recommended MLP configuration based on dataset size.
        
        Args:
            n_samples: Number of samples in dataset
            
        Returns:
            Configuration dictionary
        """
        if n_samples < 1000:
            return {
                'fast_mode': True,
                'hidden_layer_sizes': [(8,)],
                'alpha_range': [1.0, 10.0],
                'warning': 'Dataset too small - MLPs will likely overfit. Use Ridge instead.'
            }
        elif n_samples < 2000:
            return {
                'fast_mode': True,
                'hidden_layer_sizes': [(8,), (16,)],
                'alpha_range': [0.1, 1.0, 10.0],
                'max_iter': 1000,
                'message': 'Ultra-tiny MLP only. High risk of overfitting.'
            }
        elif n_samples < 5000:
            return {
                'fast_mode': False,
                'hidden_layer_sizes': [(16,), (32,), (16, 8)],
                'alpha_range': [0.01, 0.1, 1.0],
                'max_iter': 1000,
                'n_iter_no_change': 20,
                'message': 'Small MLP might work. XGBoost likely better.'
            }
        elif n_samples < 10000:
            return {
                'fast_mode': False,
                'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
                'alpha_range': [0.001, 0.01, 0.1],
                'max_iter': 2000,
                'n_iter_no_change': 30,
                'message': 'MLPs becoming viable. Can compete with tree methods.'
            }
        else:
            return {
                'fast_mode': False,
                'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100, 50)],
                'alpha_range': [0.0001, 0.001, 0.01],
                'max_iter': 2000,
                'n_iter_no_change': 50,
                'message': 'MLPs can excel here with proper tuning.'
            }