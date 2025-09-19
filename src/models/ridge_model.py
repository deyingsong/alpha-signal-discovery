from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from .base_model import BaseModel
import logging
from typing import Dict, Any

class RidgeModel(BaseModel):
    """Ridge regression model with hyperparameter tuning."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Ridge model with cross-validation for alpha selection."""
        param_grid = {'alpha': self.config.get('alpha_range', [0.1, 1.0, 10.0])}
        
        ridge = Ridge(random_state=42)
        
        # Use time series split for CV
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.config.get('cv_folds', 5))
        
        self.model = GridSearchCV(
            ridge, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        self.logger.info(f"Best alpha: {self.model.best_params_['alpha']}")
        self.logger.info(f"Best CV score: {self.model.best_score_:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature coefficients as importance measure."""
        if not self.is_fitted:
            return None
        
        coefs = self.model.best_estimator_.coef_
        return pd.Series(np.abs(coefs), index=range(len(coefs)))