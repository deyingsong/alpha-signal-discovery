import pytest
import pandas as pd
import numpy as np
from src.data.loader import DataLoader

class TestDataLoader:
    
    def setup_method(self):
        self.config = {'test': True}
        self.loader = DataLoader(self.config)
    
    def test_load_data_structure(self):
        """Test that data loading returns correct structure."""
        # This would use a mock Excel file for testing
        pass
    
    def test_validate_data_structure(self):
        """Test data structure validation."""
        # Create mock DataFrame
        df = pd.DataFrame({
            'Date': range(100),
            'Returns': np.random.normal(0, 0.01, 100),
            **{f'F{i}': np.random.normal(0, 1, 100) for i in range(1, 11)}
        })
        
        assert self.loader.validate_data_structure(df) == True
    
    def test_invalid_data_structure(self):
        """Test validation with invalid structure."""
        # Missing columns
        df = pd.DataFrame({
            'Date': range(100),
            'Returns': np.random.normal(0, 0.01, 100)
        })
        
        assert self.loader.validate_data_structure(df) == False