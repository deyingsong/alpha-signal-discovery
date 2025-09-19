"""
MLP (Multi-Layer Perceptron) Models Optimized for Small Financial Datasets
These configurations have a realistic chance of working with 1,500 samples
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# =====================================
# PYTORCH IMPLEMENTATIONS
# =====================================

class UltraTinyMLP(nn.Module):
    """
    Extremely small MLP: ~500-1000 parameters
    Best for very small datasets (<2000 samples)
    """
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()
        # Single hidden layer, very few neurons
        self.fc1 = nn.Linear(input_dim, 8)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(8, 1)
        
        # Initialize with small weights
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x.squeeze()


class TinyMLPWithBatchNorm(nn.Module):
    """
    Small MLP with batch normalization: ~2000-3000 parameters
    BatchNorm acts as regularization on small datasets
    """
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Small weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x.squeeze()


class RegularizedMLP(nn.Module):
    """
    MLP with multiple regularization techniques
    Uses: Dropout, BatchNorm, L2 weight decay, and residual connections
    """
    def __init__(self, input_dim, hidden_dims=[32, 16, 8], dropout=0.5):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout * (1 + i * 0.1)))  # Increasing dropout
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output.squeeze()


class SkipConnectionMLP(nn.Module):
    """
    MLP with skip connections - helps with gradient flow
    Good for slightly deeper networks on small data
    """
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Skip connection layer
        self.skip = nn.Linear(input_dim, hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # First layer
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout1(out)
        
        # Second layer with skip connection
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout2(out)
        
        # Add skip connection
        skip = self.skip(x)
        out = out + skip
        
        # Output layer
        out = self.fc3(out)
        return out.squeeze()


class BayesianMLP(nn.Module):
    """
    MLP with Monte Carlo Dropout for uncertainty estimation
    Keeps dropout active during inference for uncertainty
    """
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout_rate = dropout
    
    def forward(self, x, n_samples=1):
        """
        Forward pass with optional multiple samples for uncertainty
        """
        if n_samples == 1:
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x.squeeze()
        else:
            # Multiple forward passes for uncertainty estimation
            outputs = []
            for _ in range(n_samples):
                out = F.relu(self.fc1(x))
                out = F.dropout(out, p=self.dropout_rate, training=True)
                out = F.relu(self.fc2(out))
                out = F.dropout(out, p=self.dropout_rate, training=True)
                out = self.fc3(out)
                outputs.append(out)
            
            outputs = torch.stack(outputs)
            return outputs.mean(dim=0).squeeze(), outputs.std(dim=0).squeeze()


# =====================================
# SCIKIT-LEARN MLP WITH BEST CONFIGS
# =====================================

def get_sklearn_mlp_configs():
    """
    Best sklearn MLPRegressor configurations for small datasets
    """
    configs = {
        'ultra_tiny': {
            'hidden_layer_sizes': (8,),  # Single layer, 8 neurons
            'activation': 'relu',
            'alpha': 1.0,  # Strong L2 regularization
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.01,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 20,
            'random_state': 42
        },
        
        'tiny': {
            'hidden_layer_sizes': (16, 8),  # Two small layers
            'activation': 'relu',
            'alpha': 0.1,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.01,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 20,
            'random_state': 42
        },
        
        'small': {
            'hidden_layer_sizes': (32, 16, 8),  # Three layers
            'activation': 'relu',
            'alpha': 0.01,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 2000,
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 30,
            'random_state': 42
        },
        
        'regularized': {
            'hidden_layer_sizes': (20, 10),
            'activation': 'tanh',  # Sometimes better for financial data
            'alpha': 0.5,  # Heavy regularization
            'learning_rate': 'invscaling',
            'learning_rate_init': 0.01,
            'power_t': 0.5,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.25,
            'n_iter_no_change': 50,
            'momentum': 0.9,
            'random_state': 42
        }
    }
    
    return configs


# =====================================
# COMPLETE IMPLEMENTATION
# =====================================

class OptimizedMLPTrader:
    """
    Complete MLP implementation optimized for small financial datasets
    """
    
    def __init__(self, 
                 model_type='tiny',  # 'ultra_tiny', 'tiny', 'regularized', 'skip', 'bayesian'
                 use_sklearn=False,
                 input_dim=115,
                 batch_size=32,
                 learning_rate=0.001,
                 weight_decay=0.1,
                 epochs=100,
                 patience=20):
        
        self.model_type = model_type
        self.use_sklearn = use_sklearn
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        
        self.model = None
        self.scaler = StandardScaler()
        self.best_model_state = None
        
    def create_model(self):
        """Create the appropriate model"""
        if self.use_sklearn:
            configs = get_sklearn_mlp_configs()
            config = configs.get(self.model_type, configs['tiny'])
            return MLPRegressor(**config)
        else:
            if self.model_type == 'ultra_tiny':
                return UltraTinyMLP(self.input_dim, dropout=0.6)
            elif self.model_type == 'tiny':
                return TinyMLPWithBatchNorm(self.input_dim, hidden_dim=16, dropout=0.5)
            elif self.model_type == 'regularized':
                return RegularizedMLP(self.input_dim, hidden_dims=[32, 16, 8], dropout=0.5)
            elif self.model_type == 'skip':
                return SkipConnectionMLP(self.input_dim, hidden_dim=16, dropout=0.5)
            elif self.model_type == 'bayesian':
                return BayesianMLP(self.input_dim, hidden_dim=16, dropout=0.5)
            else:
                return TinyMLPWithBatchNorm(self.input_dim)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.use_sklearn:
            # Sklearn implementation
            self.model = self.create_model()
            self.model.fit(X_train_scaled, y_train)
            return self
        
        # PyTorch implementation
        if X_val is None:
            # Use last 20% as validation
            val_size = int(0.2 * len(X_train_scaled))
            X_val_scaled = X_train_scaled[-val_size:]
            y_val = y_train[-val_size:]
            X_train_scaled = X_train_scaled[:-val_size]
            y_train = y_train[:-val_size]
        else:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
        
        # Create model
        self.model = self.create_model()
        
        # Count parameters
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {self.model_type} MLP")
        print(f"Parameters: {param_count:,}")
        print(f"Data/Parameter ratio: {len(X_train_scaled)/param_count:.1f}:1")
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            # Add noise for regularization
            noise = torch.randn_like(X_train_tensor) * 0.01
            X_noisy = X_train_tensor + noise
            
            train_pred = self.model(X_noisy)
            train_loss = criterion(train_pred, y_train_tensor)
            
            # Add L1 regularization
            l1_lambda = 0.01
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            total_loss = train_loss + l1_lambda * l1_norm
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def predict(self, X, return_uncertainty=False):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        if self.use_sklearn:
            return self.model.predict(X_scaled)
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X_scaled)
        
        if return_uncertainty and isinstance(self.model, BayesianMLP):
            # Get uncertainty estimates
            with torch.no_grad():
                mean, std = self.model(X_tensor, n_samples=100)
            return mean.numpy(), std.numpy()
        else:
            with torch.no_grad():
                predictions = self.model(X_tensor).numpy()
            return predictions


# =====================================
# COMPARISON AND ANALYSIS
# =====================================

def compare_mlp_configurations(n_samples=1500, n_features=115):
    """
    Compare different MLP configurations on small data
    """
    print("="*80)
    print("MLP CONFIGURATION COMPARISON FOR SMALL FINANCIAL DATA")
    print("="*80)
    print(f"Dataset: {n_samples} samples, {n_features} features\n")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = 0.01 * X[:, :10].mean(axis=1) + 0.001 * np.random.randn(n_samples)
    
    # Train/test split
    train_size = int(0.7 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    results = {}
    
    # Test different configurations
    configs_to_test = [
        ('ultra_tiny', False),
        ('tiny', False),
        ('regularized', False),
        ('skip', False),
        ('sklearn_ultra_tiny', True),
        ('sklearn_regularized', True)
    ]
    
    for config_name, use_sklearn in configs_to_test:
        print(f"\nTesting: {config_name}")
        print("-" * 40)
        
        try:
            model_type = config_name.replace('sklearn_', '')
            
            mlp = OptimizedMLPTrader(
                model_type=model_type,
                use_sklearn=use_sklearn,
                input_dim=n_features,
                epochs=50 if not use_sklearn else None,
                patience=15
            )
            
            import time
            start = time.time()
            mlp.fit(X_train, y_train)
            train_time = time.time() - start
            
            predictions = mlp.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            results[config_name] = {
                'rmse': rmse,
                'time': train_time
            }
            
            print(f"RMSE: {rmse:.6f}")
            print(f"Training time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"Failed: {e}")
    
    # Compare with classical methods
    print("\n" + "="*60)
    print("COMPARISON WITH CLASSICAL METHODS")
    print("="*60)
    
    # Ridge
    from sklearn.linear_model import Ridge
    import time
    
    start = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_time = time.time() - start
    ridge_pred = ridge.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    
    print(f"\nRidge: RMSE = {ridge_rmse:.6f}, Time = {ridge_time:.3f}s")
    
    # XGBoost
    from xgboost import XGBRegressor
    
    start = time.time()
    xgb = XGBRegressor(n_estimators=50, max_depth=3, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - start
    xgb_pred = xgb.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    print(f"XGBoost: RMSE = {xgb_rmse:.6f}, Time = {xgb_time:.3f}s")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    all_results = {**results, 'Ridge': {'rmse': ridge_rmse, 'time': ridge_time},
                   'XGBoost': {'rmse': xgb_rmse, 'time': xgb_time}}
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['rmse'])
    
    print("\nRanking by RMSE (lower is better):")
    for i, (model, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {model:20s}: RMSE = {metrics['rmse']:.6f}, Time = {metrics['time']:.2f}s")
    
    return all_results


def parameter_analysis():
    """
    Analyze parameter counts for different architectures
    """
    print("\n" + "="*60)
    print("PARAMETER COUNT ANALYSIS")
    print("="*60)
    
    n_features = 115  # Your feature count
    n_samples = 1500  # Your sample count
    
    architectures = [
        ('Single layer (8)', [8]),
        ('Single layer (16)', [16]),
        ('Two layers (16, 8)', [16, 8]),
        ('Two layers (32, 16)', [32, 16]),
        ('Three layers (32, 16, 8)', [32, 16, 8]),
        ('Three layers (64, 32, 16)', [64, 32, 16]),
    ]
    
    print(f"\nFor {n_features} input features and {n_samples} samples:\n")
    print(f"{'Architecture':<30} {'Parameters':<12} {'Data/Param':<12} {'Viability'}")
    print("-" * 70)
    
    for name, layers in architectures:
        # Calculate parameters
        param_count = 0
        prev_size = n_features
        
        for layer_size in layers:
            param_count += (prev_size + 1) * layer_size  # +1 for bias
            prev_size = layer_size
        
        param_count += (prev_size + 1) * 1  # Output layer
        
        ratio = n_samples / param_count
        
        if ratio > 30:
            viability = "✅ Good"
        elif ratio > 10:
            viability = "⚠️  Risky"
        else:
            viability = "❌ Will overfit"
        
        print(f"{name:<30} {param_count:<12,} {ratio:<12.1f} {viability}")
    
    print(f"\n💡 Recommendation: Use architectures with <5,000 parameters")
    print(f"   Best options: Single layer (8-16) or Two layers (16, 8)")


if __name__ == "__main__":
    print("\n🧠 MLP Analysis for Small Financial Datasets\n")
    
    # Run parameter analysis
    parameter_analysis()
    
    # Run comparison
    print("\n" + "="*80)
    print("Running MLP comparison (this may take a minute)...")
    print("="*80)
    
    results = compare_mlp_configurations(n_samples=1500)
    
    print("\n" + "="*80)
    print("✅ FINAL RECOMMENDATIONS")
    print("="*80)
    print("""
    For 1,500 samples in quantitative trading:
    
    1. IF you must use neural networks, use MLP (not RNN/LSTM/Transformer)
    2. Best MLP configs:
       - Ultra-tiny: (8,) single layer - safest option
       - Tiny: (16, 8) two layers - good balance
       - Sklearn MLPRegressor with early_stopping=True
    
    3. Critical settings:
       - Dropout: 0.5-0.7 (very high)
       - Weight decay: 0.01-0.1 (heavy L2)
       - Early stopping: Essential (patience=10-20)
       - Batch normalization: Helps a lot
       - Learning rate: Start at 0.001, use scheduler
    
    4. Expected performance:
       - MLPs might match Ridge/XGBoost if perfectly tuned
       - Training will be 10-50x slower than Ridge
       - More prone to overfitting than tree-based methods
    
    5. When MLPs become worthwhile:
       - 5,000+ samples: MLPs become competitive
       - 10,000+ samples: MLPs can sometimes win
       - 50,000+ samples: Deep MLPs excel
    
    Bottom line: For your 1,500 samples, a tiny MLP (8-16 neurons)
    with heavy regularization is the only neural network worth trying.
    """)