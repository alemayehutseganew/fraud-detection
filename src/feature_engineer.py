"""
Feature engineering utilities for fraud detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Engineer features for fraud detection."""
    
    def __init__(self):
        """Initialize feature engineer."""
        pass
    
    def create_all_features(self, df):
        """Create all engineered features."""
        df_features = df.copy()
        
        # Basic features
        if 'purchase_value' in df_features.columns:
            df_features['purchase_value_log'] = np.log1p(df_features['purchase_value'])
        
        # Time-based features
        if 'purchase_time' in df_features.columns:
            df_features['hour_of_day'] = df_features['purchase_time'].dt.hour
            df_features['day_of_week'] = df_features['purchase_time'].dt.dayofweek
            df_features['month'] = df_features['purchase_time'].dt.month
        
        # Categorical encoding
        if 'source' in df_features.columns:
            df_features['source_encoded'] = df_features['source'].astype('category').cat.codes
        
        if 'browser' in df_features.columns:
            df_features['browser_encoded'] = df_features['browser'].astype('category').cat.codes
        
        if 'sex' in df_features.columns:
            df_features['sex_encoded'] = df_features['sex'].map({'M': 1, 'F': 0})
        
        # Country features
        if 'country' in df_features.columns:
            df_features['country_encoded'] = df_features['country'].astype('category').cat.codes
        
        print(f"Created {len(df_features.columns) - len(df.columns)} new features")
        
        return df_features