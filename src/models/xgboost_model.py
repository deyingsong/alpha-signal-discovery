import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from .base_model import BaseModel
import logging
from typing import Dict, Any

class XGBoostModel(BaseModel):
    """XGBoost model with hyperparameter tuning."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost with hyperparameter tuning."""
        param_grid = {
            'n_estimators': self.config.get('n_estimators_range', [100, 200]),
            'max_depth': self.config.get('max_depth_range', [3, 5]),
            'learning_rate': self.config.get('learning_rate_range', [0.05, 0.1]),
            'reg_alpha': self.config.get('reg_alpha_range', [0, 0.1]),
            'reg_lambda': self.config.get('reg_lambda_range', [1, 1.5])
        }
        
        xgb_model = xgb.XGBRegressor(
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        self.model = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.logger.info(f"Best params: {self.model.best_params_}")
        self.logger.info(f"Best CV score: {self.model.best_score_:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from XGBoost."""
        if not self.is_fitted:
            return None
        
        importance = self.model.best_estimator_.feature_importances_
        return pd.Series(importance, index=range(len(importance)))