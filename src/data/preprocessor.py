import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import logging

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.logger = logging.getLogger(__name__)
    
    def align_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features with forward-shifted returns to prevent lookahead bias."""
        # Features at time t predict returns at time t+1
        features = df.drop(['Date', 'Returns'], axis=1).iloc[:-1]  # Remove last row
        targets = df['Returns'].iloc[1:].reset_index(drop=True)    # Shift returns forward
        
        # Reset index to ensure alignment
        features = features.reset_index(drop=True)
        
        self.logger.info(f"Aligned {len(features)} feature-target pairs")
        return features, targets
    
    def split_train_test(self, X: pd.DataFrame, y: pd.Series, 
                        train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                          pd.Series, pd.Series]:
        """Split data chronologically."""
        split_idx = int(len(X) * train_ratio)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def analyze_signal_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze statistical properties of signals."""
        signal_cols = [f'F{i}' for i in range(1, 11)]
        
        stats = pd.DataFrame({
            'Mean': df[signal_cols].mean(),
            'Std': df[signal_cols].std(),
            'Skewness': df[signal_cols].skew(),
            'Kurtosis': df[signal_cols].kurtosis() + 3,  # Add 3 for standard kurtosis
            'Min': df[signal_cols].min(),
            'Max': df[signal_cols].max()
        })
        
        # Check for identical properties
        if stats.nunique().max() == 1:
            self.logger.info("All signals have identical statistical properties")
        
        return stats
    
    # Add outlier detection and treatment
    def winsorize_features(df, lower_percentile=0.01, upper_percentile=0.99):
        """Winsorize extreme outliers"""
        for col in [f'F{i}' for i in range(1, 11)]:
            lower_bound = df[col].quantile(lower_percentile)
            upper_bound = df[col].quantile(upper_percentile)
            df[col] = df[col].clip(lower_bound, upper_bound)
        return df