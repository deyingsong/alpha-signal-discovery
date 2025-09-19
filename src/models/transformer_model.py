import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel
import logging
from torch.utils.data import DataLoader, TensorDataset
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerNet(nn.Module):
    """Transformer architecture for time series prediction."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, 
                 dropout: float = 0.1, max_seq_length: int = 100):
        super(TransformerNet, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Global attention pooling
        self.attention_weights = nn.Linear(d_model, 1)
        
        # Output layers with residual connections
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.layer_norm1 = nn.LayerNorm(d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.layer_norm2 = nn.LayerNorm(d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, 1)
        
        # Activation
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Create attention mask for padding (if needed)
        # For simplicity, we assume no padding in this implementation
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x)
        # transformer_out shape: (batch_size, seq_len, d_model)
        
        # Global attention pooling
        attention_scores = self.attention_weights(transformer_out)
        attention_scores = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of transformer outputs
        context = torch.sum(attention_scores * transformer_out, dim=1)
        # context shape: (batch_size, d_model)
        
        # Pass through output layers with residual connections
        out = self.dropout(context)
        
        # First layer
        out1 = self.fc1(out)
        out1 = self.layer_norm1(out1)
        out1 = self.gelu(out1)
        out1 = self.dropout(out1)
        
        # Second layer
        out2 = self.fc2(out1)
        out2 = self.layer_norm2(out2)
        out2 = self.gelu(out2)
        out2 = self.dropout(out2)
        
        # Final output
        out = self.fc3(out2)
        
        return out.squeeze()

class TransformerModel(BaseModel):
    """Transformer model for financial time series prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.sequence_length = config.get('sequence_length', 40)
        self.best_model_state = None
        
    def create_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple:
        """Create sequences for Transformer training."""
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X.iloc[i-self.sequence_length:i].values)
            if y is not None:
                y_seq.append(y.iloc[i])
        
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq
    
    def create_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to enhance transformer performance."""
        X_enhanced = X.copy()
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            for col in X.columns[:10]:  # Assuming first 10 are base features
                X_enhanced[f'{col}_roll_mean_{window}'] = X[col].rolling(window).mean()
                X_enhanced[f'{col}_roll_std_{window}'] = X[col].rolling(window).std()
        
        # Fill NaN values from rolling
        X_enhanced = X_enhanced.fillna(0)
        
        return X_enhanced
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Transformer model."""
        # Add temporal features if configured
        if self.config.get('use_temporal_features', False):
            X = self.create_temporal_features(X)
        
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Convert back to DataFrame for sequence creation
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        y_scaled_series = pd.Series(y_scaled)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled_df, y_scaled_series)
        
        # Split into train and validation
        val_split = self.config.get('validation_split', 0.1)
        val_size = int(len(X_seq) * val_split)
        
        X_train = X_seq[:-val_size] if val_size > 0 else X_seq
        y_train = y_seq[:-val_size] if val_size > 0 else y_seq
        X_val = X_seq[-val_size:] if val_size > 0 else None
        y_val = y_seq[-val_size:] if val_size > 0 else None
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loader with larger batch size for transformer
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = self.config.get('batch_size', 64)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=0, pin_memory=True)
        
        # Initialize model
        input_size = X.shape[1]
        d_model = self.config.get('d_model', 128)
        nhead = self.config.get('nhead', 8)
        num_encoder_layers = self.config.get('num_encoder_layers', 4)
        dim_feedforward = self.config.get('dim_feedforward', 512)
        dropout = self.config.get('dropout', 0.2)
        
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = (d_model // nhead) * nhead
            self.logger.warning(f"Adjusted d_model to {d_model} to be divisible by nhead")
        
        self.model = TransformerNet(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=self.sequence_length
        ).to(self.device)
        
        # Training parameters
        learning_rate = self.config.get('learning_rate', 0.0001)
        weight_decay = self.config.get('weight_decay', 0.01)
        epochs = self.config.get('epochs', 200)
        patience = self.config.get('patience', 20)
        warmup_epochs = self.config.get('warmup_epochs', 10)
        
        # Use AdamW optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay, betas=(0.9, 0.999))
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, verbose=False
        )
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Add label smoothing regularization
                if self.config.get('label_smoothing', 0.0) > 0:
                    smoothing = self.config.get('label_smoothing', 0.1)
                    smooth_loss = torch.mean((outputs - batch_y.mean()) ** 2)
                    loss = (1 - smoothing) * loss + smoothing * smooth_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(loader)
            
            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                current_loss = val_loss
            else:
                current_loss = avg_train_loss
            
            # Update learning rate
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(current_loss)
            
            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                               f"Val/Best Loss: {current_loss:.6f}, LR: {current_lr:.6f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        self.logger.info(f"Transformer training completed. Best loss: {best_loss:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Transformer."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        
        # Add temporal features if configured
        if self.config.get('use_temporal_features', False):
            X = self.create_temporal_features(X)
        
        # Scale features
        X_scaled = self.scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled_df)
        
        # Handle case where we don't have enough history
        if len(X_seq) == 0:
            padding_needed = self.sequence_length - len(X)
            if padding_needed > 0:
                padding = np.zeros((padding_needed, X.shape[1]))
                X_padded = np.vstack([padding, X_scaled])
                X_seq = np.array([X_padded])
            else:
                X_seq = np.array([X_scaled[-self.sequence_length:]])
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions
        predictions = self.scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        # Pad predictions to match input length
        if len(predictions) < len(X):
            padding = np.zeros(len(X) - len(predictions))
            predictions = np.concatenate([padding, predictions])
        
        return predictions