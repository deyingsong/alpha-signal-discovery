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

class RNNNet(nn.Module):
    """RNN architecture for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float = 0.2, bidirectional: bool = False):
        super(RNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, 
                        self.hidden_size).to(x.device)
        
        # RNN forward pass
        out, _ = self.rnn(x, h0)
        
        # Take the last output
        if self.bidirectional:
            out = out[:, -1, :]  # (batch_size, hidden_size * 2)
        else:
            out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # Pass through fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out.squeeze()

class RNNModel(BaseModel):
    """RNN model for financial time series prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.sequence_length = config.get('sequence_length', 20)
        self.best_model_state = None
        
    def create_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple:
        """Create sequences for RNN training."""
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train RNN model."""
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Convert back to DataFrame for sequence creation
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        y_scaled_series = pd.Series(y_scaled)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled_df, y_scaled_series)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = self.config.get('batch_size', 32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_size = X.shape[1]
        hidden_size = self.config.get('hidden_size', 64)
        num_layers = self.config.get('num_layers', 2)
        dropout = self.config.get('dropout', 0.3)
        bidirectional = self.config.get('bidirectional', False)
        
        self.model = RNNNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Training parameters
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.01)
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, 
                              weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        self.logger.info(f"RNN training completed. Best loss: {best_loss:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with RNN."""
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
            # Pad with zeros if needed
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