from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}