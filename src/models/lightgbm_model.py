import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from .base_model import BaseModel
import logging
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

class LightGBMModel(BaseModel):
    """LightGBM model with hyperparameter tuning for financial time series prediction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LightGBM with hyperparameter tuning using time series cross-validation."""
        
        # Store feature names for later use
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': self.config.get('n_estimators_range', [100, 200, 300, 500]),
            'max_depth': self.config.get('max_depth_range', [-1, 3, 5, 7, 10]),  # -1 means no limit in LightGBM
            'learning_rate': self.config.get('learning_rate_range', [0.01, 0.05, 0.1, 0.2]),
            'num_leaves': self.config.get('num_leaves_range', [31, 50, 100, 200]),
            'min_child_samples': self.config.get('min_child_samples_range', [20, 30, 50]),
            'subsample': self.config.get('subsample_range', [0.6, 0.8, 1.0]),
            'colsample_bytree': self.config.get('colsample_bytree_range', [0.6, 0.8, 1.0]),
            'reg_alpha': self.config.get('reg_alpha_range', [0, 0.01, 0.1, 1.0]),
            'reg_lambda': self.config.get('reg_lambda_range', [0, 0.01, 0.1, 1.0])
        }
        
        # Initialize base LightGBM regressor with fixed parameters
        lgb_base = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1),
            importance_type='gain',
            verbosity=-1,  # Suppress LightGBM warnings
            force_col_wise=True,  # Optimization for many features
            min_gain_to_split=self.config.get('min_gain_to_split', 0.01),
            min_data_in_bin=self.config.get('min_data_in_bin', 3),
            feature_fraction_bynode=self.config.get('feature_fraction_bynode', 1.0),
            lambda_l1=0,  # Will be overridden by reg_alpha in param_grid
            lambda_l2=0   # Will be overridden by reg_lambda in param_grid
        )
        
        # Use TimeSeriesSplit for cross-validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.config.get('cv_folds', 3))
        
        # Grid search with time series cross-validation
        self.model = GridSearchCV(
            lgb_base,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,  # Avoid nested parallelism
            verbose=0
        )
        
        # Fit the model
        self.model.fit(
            X, y,
            callbacks=[lgb.log_evaluation(0)]  # Suppress iteration logs
        )
        
        self.is_fitted = True
        
        # Log best parameters and score
        self.logger.info(f"Best params: {self.model.best_params_}")
        self.logger.info(f"Best CV score: {self.model.best_score_:.6f}")
        
        # Store feature importance for analysis
        self._store_feature_importance()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # LightGBM can handle both DataFrame and numpy array
        predictions = self.model.predict(X)
        
        return predictions
    
    def _store_feature_importance(self) -> None:
        """Store feature importance from the best estimator."""
        if hasattr(self.model, 'best_estimator_'):
            self.feature_importance_ = self.model.best_estimator_.feature_importances_
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from LightGBM model."""
        if not self.is_fitted:
            return None
        
        if hasattr(self, 'feature_importance_'):
            importance = self.feature_importance_
            
            # Create a Series with feature names if available
            if self.feature_names is not None:
                return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
            else:
                return pd.Series(importance, index=range(len(importance)))
        
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get the best parameters found during grid search."""
        if self.model and hasattr(self.model, 'best_params_'):
            return self.model.best_params_
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for analysis."""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        info = {
            "best_params": self.get_params(),
            "best_cv_score": self.model.best_score_ if hasattr(self.model, 'best_score_') else None,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "model_type": "LightGBM"
        }
        
        # Add tree-specific information if available
        if hasattr(self.model, 'best_estimator_'):
            best_model = self.model.best_estimator_
            info.update({
                "n_estimators_used": best_model.n_estimators_,
                "n_features_used": best_model.n_features_in_ if hasattr(best_model, 'n_features_in_') else None,
                "objective": best_model.objective_,
                "boosting_type": best_model.boosting_type
            })
        
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a previously saved model from disk."""
        import joblib
        self.model = joblib.load(filepath)
        self.is_fitted = True
        self.logger.info(f"Model loaded from {filepath}")
        
        # Restore feature importance if available
        if hasattr(self.model, 'best_estimator_'):
            self._store_feature_importance()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities if applicable.
        For regression, this returns the same as predict().
        """
        return self.predict(X)
    
    def get_n_iterations(self) -> Optional[int]:
        """Get the actual number of boosting iterations used."""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'best_estimator_'):
            # LightGBM stores the actual number of trees used
            return self.model.best_estimator_.n_estimators_
        
        return None
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """Plot feature importance using LightGBM's built-in plotting."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature importance")
        
        if hasattr(self.model, 'best_estimator_'):
            import matplotlib.pyplot as plt
            
            # Get feature importance
            importance = self.get_feature_importance()
            
            if importance is not None and len(importance) > 0:
                # Select top N features
                top_features = importance.head(top_n)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features.plot(kind='barh', ax=ax)
                ax.set_xlabel('Feature Importance (Gain)')
                ax.set_title(f'Top {top_n} Feature Importances - LightGBM Model')
                plt.tight_layout()
                plt.show()
            else:
                self.logger.warning("No feature importance data available")
    
    def get_trees_as_dataframe(self) -> Optional[pd.DataFrame]:
        """Export the model's trees structure as a DataFrame for analysis."""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'best_estimator_'):
            # LightGBM can export its model structure
            booster = self.model.best_estimator_.booster_
            
            if booster is not None:
                # Get tree structure as a string and parse it
                tree_info = booster.dump_model()
                
                # Return basic tree statistics
                return pd.DataFrame({
                    'num_trees': [tree_info.get('num_trees', 0)],
                    'num_leaves': [tree_info.get('num_leaves', 0)],
                    'max_depth': [tree_info.get('max_depth', 0)]
                })
        
        return None