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

class LSTMNet(nn.Module):
    """LSTM architecture for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float = 0.2, bidirectional: bool = False):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * self.num_directions)
        
        # Dropout and fully connected layers
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, 
                        self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, 
                        self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context shape: (batch_size, hidden_size * num_directions)
        
        # Apply batch normalization
        context = self.batch_norm(context)
        
        # Pass through fully connected layers
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out.squeeze()

class LSTMModel(BaseModel):
    """LSTM model for financial time series prediction with attention mechanism."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.sequence_length = config.get('sequence_length', 30)
        self.best_model_state = None
        
    def create_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple:
        """Create sequences for LSTM training with overlapping windows."""
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
    
    def augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation for time series."""
        augmented_X = []
        augmented_y = []
        
        # Original data
        augmented_X.append(X)
        augmented_y.append(y)
        
        # Add noise
        noise_level = self.config.get('noise_level', 0.01)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, X.shape)
            augmented_X.append(X + noise)
            augmented_y.append(y)
        
        # Time warping (slight scaling of sequences)
        if self.config.get('time_warping', True):
            scale_factor = np.random.uniform(0.95, 1.05)
            warped_X = X * scale_factor
            augmented_X.append(warped_X)
            augmented_y.append(y)
        
        return np.concatenate(augmented_X), np.concatenate(augmented_y)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train LSTM model with advanced techniques."""
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Convert back to DataFrame for sequence creation
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        y_scaled_series = pd.Series(y_scaled)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled_df, y_scaled_series)
        
        # Apply data augmentation
        if self.config.get('augmentation', False):
            X_seq, y_seq = self.augment_data(X_seq, y_seq)
        
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
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = self.config.get('batch_size', 32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_size = X.shape[1]
        hidden_size = self.config.get('hidden_size', 128)
        num_layers = self.config.get('num_layers', 3)
        dropout = self.config.get('dropout', 0.3)
        bidirectional = self.config.get('bidirectional', True)
        
        self.model = LSTMNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Training parameters
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.01)
        epochs = self.config.get('epochs', 150)
        patience = self.config.get('patience', 15)
        
        # Use AdamW optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay)
        
        # Loss function with possible custom weighting
        criterion = nn.MSELoss()
        
        # Learning rate schedulers
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Cosine annealing for warm restarts
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        # Training loop with early stopping
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
                
                # Add L2 regularization to loss
                l2_lambda = self.config.get('l2_lambda', 0.001)
                l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss = loss + l2_lambda * l2_norm
                
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
                
                # Use validation loss for early stopping
                current_loss = val_loss
            else:
                current_loss = avg_train_loss
            
            # Update learning rate
            scheduler.step(current_loss)
            cosine_scheduler.step()
            
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
                self.logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, "
                               f"Val/Best Loss: {current_loss:.6f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        self.logger.info(f"LSTM training completed. Best loss: {best_loss:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        
        # Scale features
        X_scaled = self.scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled_df)
        
        # Handle case where we don't have enough history for sequences
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
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Extract attention weights as feature importance (if applicable)."""
        # This is a simplified version - in practice, you'd analyze attention weights
        return None