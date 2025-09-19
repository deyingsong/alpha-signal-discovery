import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from pathlib import Path

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, float]:
        """Load data from Excel file and extract transaction costs."""
        try:
            # Read Excel file
            df = pd.read_excel(filepath, sheet_name='DATA')
            
            # Extract transaction cost from first row
            transaction_cost = df.iloc[0, 1]  # Should be 0.005
            
            # Find header row (contains 'Date')
            header_row = None
            for i, row in df.iterrows():
                if 'Date' in str(row.iloc[0]):
                    header_row = i
                    break
            
            if header_row is None:
                raise ValueError("Could not find header row containing 'Date'")
            
            # Extract actual data
            headers = df.iloc[header_row].values
            data = df.iloc[header_row + 1:].values
            
            # Create clean DataFrame
            clean_df = pd.DataFrame(data, columns=headers)
            clean_df = clean_df.dropna().reset_index(drop=True)
            
            # Convert to appropriate data types
            clean_df['Date'] = clean_df['Date'].astype(int)
            for col in clean_df.columns[1:]:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
            self.logger.info(f"Loaded {len(clean_df)} observations")
            self.logger.info(f"Transaction cost: {transaction_cost}")
            
            return clean_df, transaction_cost
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data_structure(self, df: pd.DataFrame) -> bool:
        """Validate the expected data structure."""
        expected_columns = ['Date', 'Returns'] + [f'F{i}' for i in range(1, 11)]
        
        if list(df.columns) != expected_columns:
            self.logger.error(f"Unexpected columns: {list(df.columns)}")
            return False
        
        if len(df) != 1500:
            self.logger.warning(f"Expected 1500 rows, got {len(df)}")
        
        return True