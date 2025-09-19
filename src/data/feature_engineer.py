import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from itertools import combinations
import logging

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(
            degree=config.get('polynomial_degree', 2), 
            include_bias=False
        )
        self.logger = logging.getLogger(__name__)
    
    def create_lagged_features(self, X: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Create lagged versions of features."""
        feature_cols = [f'F{i}' for i in range(1, 11)]
        lagged_features = []
        
        for lag in lags:
            for col in feature_cols:
                lagged_col = f"{col}_lag_{lag}"
                lagged_features.append(X[col].shift(lag).fillna(0))
        
        lagged_df = pd.concat(lagged_features, axis=1)
        lagged_df.columns = [f"{col}_lag_{lag}" for lag in lags for col in feature_cols]
        
        self.logger.info(f"Created {len(lagged_df.columns)} lagged features")
        return lagged_df
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create pairwise interaction features."""
        feature_cols = [f'F{i}' for i in range(1, 11)]
        interactions = []
        interaction_names = []
        
        for col1, col2 in combinations(feature_cols, 2):
            interaction = X[col1] * X[col2]
            interactions.append(interaction)
            interaction_names.append(f"{col1}_{col2}_interaction")
        
        interaction_df = pd.concat(interactions, axis=1)
        interaction_df.columns = interaction_names
        
        self.logger.info(f"Created {len(interaction_df.columns)} interaction features")
        return interaction_df
    
    def create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features (squared terms)."""
        feature_cols = [f'F{i}' for i in range(1, 11)]
        poly_features = []
        
        for col in feature_cols:
            squared_feature = X[col] ** 2
            poly_features.append(squared_feature)
        
        poly_df = pd.concat(poly_features, axis=1)
        poly_df.columns = [f"{col}_squared" for col in feature_cols]
        
        self.logger.info(f"Created {len(poly_df.columns)} polynomial features")
        return poly_df
    
    def engineer_all_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        # Start with original features
        engineered_X = X.copy()
        
        # Add lagged features
        if self.config.get('lag_periods'):
            lagged_features = self.create_lagged_features(X, self.config['lag_periods'])
            engineered_X = pd.concat([engineered_X, lagged_features], axis=1)
        
        # Add interaction features
        if self.config.get('interaction_features', True):
            interaction_features = self.create_interaction_features(X)
            engineered_X = pd.concat([engineered_X, interaction_features], axis=1)
        
        # Add polynomial features
        if self.config.get('polynomial_degree', 2) > 1:
            poly_features = self.create_polynomial_features(X)
            engineered_X = pd.concat([engineered_X, poly_features], axis=1)
        
        # Remove any NaN values from lagging
        engineered_X = engineered_X.fillna(0)
        
        self.logger.info(f"Total engineered features: {engineered_X.shape[1]}")
        return engineered_X
    
    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """Fit scaler on training data only."""
        self.scaler.fit(X_train)
        self.logger.info("Fitted feature scaler on training data")
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled
    
    def create_technical_features(df):
        """Add financial-specific features"""
        # Moving averages
        for window in [5, 10, 20]:
            for col in [f'F{i}' for i in range(1, 11)]:
                df[f'{col}_ma_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window).std()
        
        # Momentum features
        for period in [3, 5, 10]:
            for col in [f'F{i}' for i in range(1, 11)]:
                df[f'{col}_momentum_{period}'] = df[col] / df[col].shift(period) - 1
        
        # Volatility features
        df['returns_volatility_10'] = df['Returns'].rolling(10).std()
        
        return df
    
    def create_cross_signal_features(df):
        """Create features that capture relationships between signals"""
        signal_cols = [f'F{i}' for i in range(1, 11)]
        
        # Signal strength (sum of absolute values)
        df['signal_strength'] = df[signal_cols].abs().sum(axis=1)
        
        # Signal direction consensus
        df['signal_consensus'] = df[signal_cols].apply(lambda x: (x > 0).sum() - (x < 0).sum(), axis=1)
        
        # PCA components (since signals appear pre-processed)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(df[signal_cols])
        for i in range(5):
            df[f'pca_component_{i}'] = pca_features[:, i]
        
        return df