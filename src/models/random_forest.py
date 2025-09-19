from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from .base_model import BaseModel
import logging
from typing import Dict, Any

class RandomForestModel(BaseModel):
    """Random Forest model with hyperparameter tuning."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Random Forest with hyperparameter tuning."""
        param_grid = {
            'n_estimators': self.config.get('n_estimators_range', [100, 200]),
            'max_depth': self.config.get('max_depth_range', [5, 10, None]),
            'min_samples_split': self.config.get('min_samples_split_range', [2, 5])
        }
        
        rf = RandomForestRegressor(
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits for complex models
        
        self.model = GridSearchCV(
            rf, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=1  # Avoid nested parallelism
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
        """Get feature importance from Random Forest."""
        if not self.is_fitted:
            return None
        
        importance = self.model.best_estimator_.feature_importances_
        return pd.Series(importance, index=range(len(importance)))
