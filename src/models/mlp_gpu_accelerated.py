import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel
import logging
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader, TensorDataset

class MLPGPUModel(BaseModel):
    """GPU-accelerated Multi-Layer Perceptron using PyTorch with Apple Silicon support."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Detect available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using Apple Silicon MPS for GPU acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info("Using CUDA for GPU acceleration")
        else:
            self.device = torch.device("cpu")
            self.logger.info("No GPU available, using CPU")
        
        self.scaler = StandardScaler()
        self.model = None
        self.best_model_state = None
        
    class MLPNetwork(nn.Module):
        """Advanced MLP architecture with regularization."""
        
        def __init__(self, input_dim: int, hidden_dims: list = None, 
                    dropout_rate: float = 0.3, use_batch_norm: bool = True):
            super().__init__()
            
            if hidden_dims is None:
                # Default architecture
                hidden_dims = [256, 128, 64, 32]
            
            self.use_batch_norm = use_batch_norm
            layers = []
            prev_dim = input_dim
            
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                
                layers.append(nn.ReLU())
                
                if i < len(hidden_dims) - 1:  # Don't add dropout after last hidden layer
                    layers.append(nn.Dropout(dropout_rate))
                
                prev_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(prev_dim, 1))
            
            self.network = nn.Sequential(*layers)
            
            # Initialize weights
            self._initialize_weights()
        
        def _initialize_weights(self):
            """Xavier/He initialization for better convergence."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        
        def forward(self, x):
            return self.network(x)
    
    def create_dataloaders(self, X: np.ndarray, y: np.ndarray, 
                          batch_size: int = 32, validation_split: float = 0.1) -> Tuple:
        """Create PyTorch dataloaders with optional validation split."""
        
        # Create validation split
        val_size = int(len(X) * validation_split)
        
        X_train = X[:-val_size] if val_size > 0 else X
        y_train = y[:-val_size] if val_size > 0 else y
        X_val = X[-val_size:] if val_size > 0 else None
        y_val = y[-val_size:] if val_size > 0 else None
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,  # Don't shuffle time series
            pin_memory=True if self.device != torch.device("cpu") else False
        )
        
        val_loader = None
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=True if self.device != torch.device("cpu") else False
            )
        
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> float:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit MLP model with GPU acceleration."""
        
        # Convert to numpy and scale
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        X_scaled = self.scaler.fit_transform(X_np)
        
        # Hyperparameter search using cross-validation
        hidden_dims_options = [
            [128, 64, 32],
            [256, 128, 64, 32],
            [512, 256, 128, 64]
        ]
        learning_rates = [0.001, 0.0005, 0.0001]
        
        best_val_loss = float('inf')
        best_config = None
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for hidden_dims in hidden_dims_options:
            for lr in learning_rates:
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train_cv = X_scaled[train_idx]
                    y_train_cv = y_np[train_idx]
                    X_val_cv = X_scaled[val_idx]
                    y_val_cv = y_np[val_idx]
                    
                    # Create model
                    model = self.MLPNetwork(
                        input_dim=X_scaled.shape[1],
                        hidden_dims=hidden_dims,
                        dropout_rate=0.2,
                        use_batch_norm=True
                    ).to(self.device)
                    
                    # Setup training
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.AdamW(
                        model.parameters(), 
                        lr=lr,
                        weight_decay=0.01
                    )
                    
                    # Quick training for hyperparameter search
                    train_loader, _ = self.create_dataloaders(
                        X_train_cv, y_train_cv, 
                        batch_size=64, 
                        validation_split=0
                    )
                    
                    for epoch in range(20):  # Quick training
                        self.train_epoch(model, train_loader, optimizer, criterion)
                    
                    # Validate
                    model.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_val_cv).to(self.device)
                        y_val_tensor = torch.FloatTensor(y_val_cv).reshape(-1, 1).to(self.device)
                        val_pred = model(X_val_tensor)
                        val_loss = criterion(val_pred, y_val_tensor).item()
                        cv_scores.append(val_loss)
                
                avg_cv_score = np.mean(cv_scores)
                if avg_cv_score < best_val_loss:
                    best_val_loss = avg_cv_score
                    best_config = {'hidden_dims': hidden_dims, 'lr': lr}
        
        self.logger.info(f"Best config: {best_config}")
        self.logger.info(f"Best CV score: {-best_val_loss:.6f}")
        
        # Train final model with best configuration
        self.model = self.MLPNetwork(
            input_dim=X_scaled.shape[1],
            hidden_dims=best_config['hidden_dims'],
            dropout_rate=0.2,
            use_batch_norm=True
        ).to(self.device)
        
        # Setup final training
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=best_config['lr'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(
            X_scaled, y_np, 
            batch_size=32, 
            validation_split=0.1
        )
        
        # Training with early stopping
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(100):
            train_loss = self.train_epoch(self.model, train_loader, optimizer, criterion)
            
            if val_loader:
                val_loss = self.validate_epoch(self.model, val_loader, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            scheduler.step()
            
            if epoch % 20 == 0:
                self.logger.debug(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using GPU."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_np)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance using gradient-based method."""
        if not self.is_fitted:
            return None
        
        # Use input gradients as importance measure
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.zeros(1, self.model.network[0].in_features).to(self.device)
        dummy_input.requires_grad = True
        
        # Forward pass
        output = self.model(dummy_input)
        
        # Backward pass
        output.backward()
        
        # Get gradients
        importance = dummy_input.grad.abs().mean(dim=0).cpu().numpy()
        
        return pd.Series(importance, index=range(len(importance)))