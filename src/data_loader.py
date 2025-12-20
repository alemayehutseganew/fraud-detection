"""
Data loading utilities for fraud detection project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and manage fraud detection datasets."""
    
    def __init__(self, data_dir='data/raw'):
        """Initialize data loader with data directory."""
        self.data_dir = Path(data_dir)
        
    def load_fraud_data(self, file_name='Fraud_Data.csv'):
        """Load e-commerce fraud dataset."""
        file_path = self.data_dir / file_name
        df = pd.read_csv(file_path)
        print(f"Loaded Fraud Data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_ip_country_data(self, file_name='IpAddress_to_Country.csv'):
        """Load IP to country mapping dataset."""
        file_path = self.data_dir / file_name
        df = pd.read_csv(file_path)
        print(f"Loaded IP Country Data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_creditcard_data(self, file_name='creditcard.csv'):
        """Load credit card fraud dataset."""
        file_path = self.data_dir / file_name
        df = pd.read_csv(file_path)
        print(f"Loaded Credit Card Data: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_all_data(self):
        """Load all datasets."""
        fraud_data = self.load_fraud_data()
        ip_data = self.load_ip_country_data()
        credit_data = self.load_creditcard_data()
        
        return {
            'fraud_data': fraud_data,
            'ip_data': ip_data,
            'credit_data': credit_data
        }
    
    def get_data_info(self, df):
        """Get basic information about a dataframe."""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return info