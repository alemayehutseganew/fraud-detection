"""
Tests for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import FraudDataPreprocessor


class TestFraudDataPreprocessor:
    """Test FraudDataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample fraud data for testing."""
        data = {
            'user_id': [1, 2, 3, 4, 5],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', 
                           '2023-01-01 12:00:00', '2023-01-01 13:00:00',
                           '2023-01-01 14:00:00'],
            'purchase_time': ['2023-01-01 10:30:00', '2023-01-01 11:15:00',
                             '2023-01-01 12:45:00', '2023-01-01 13:20:00',
                             '2023-01-01 14:10:00'],
            'purchase_value': [100.0, 200.0, 150.0, 300.0, 250.0],
            'device_id': ['A', 'B', 'C', 'D', 'E'],
            'source': ['SEO', 'Ads', 'Direct', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Firefox', 'Chrome', 'Safari', 'Chrome'],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'age': [25, 30, 35, 28, 32],
            'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3',
                          '192.168.1.4', '192.168.1.5'],
            'class': [0, 1, 0, 0, 1]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_ip_data(self):
        """Create sample IP country data for testing."""
        data = {
            'lower_bound_ip_address': ['192.168.1.0', '192.168.2.0'],
            'upper_bound_ip_address': ['192.168.1.255', '192.168.2.255'],
            'country': ['CountryA', 'CountryB']
        }
        return pd.DataFrame(data)
    
    def test_clean_data(self, sample_data):
        """Test data cleaning functionality."""
        preprocessor = FraudDataPreprocessor()
        cleaned_data = preprocessor.clean_data(sample_data)
        
        # Check shape
        assert cleaned_data.shape == sample_data.shape
        
        # Check datetime conversion
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['signup_time'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['purchase_time'])
        
        # Check no missing values in numeric columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert cleaned_data[col].isnull().sum() == 0
    
    def test_convert_ip_to_int(self):
        """Test IP address to integer conversion."""
        preprocessor = FraudDataPreprocessor()
        
        # Test valid IP
        ip_int = preprocessor.convert_ip_to_int('192.168.1.1')
        assert isinstance(ip_int, int)
        assert ip_int == 3232235777
        
        # Test another IP
        ip_int = preprocessor.convert_ip_to_int('10.0.0.1')
        assert ip_int == 167772161
    
    def test_create_time_features(self, sample_data):
        """Test time feature creation."""
        preprocessor = FraudDataPreprocessor()
        
        # First clean the data to convert timestamps
        cleaned_data = preprocessor.clean_data(sample_data)
        
        # Create time features
        features_data = preprocessor.create_time_features(cleaned_data)
        
        # Check new features exist
        assert 'purchase_hour' in features_data.columns
        assert 'time_since_signup' in features_data.columns
        assert 'same_day_purchase' in features_data.columns
        
        # Check time_since_signup calculation
        for idx, row in features_data.iterrows():
            expected_hours = (row['purchase_time'] - row['signup_time']).total_seconds() / 3600
            assert abs(row['time_since_signup'] - expected_hours) < 0.1
    
    def test_merge_with_ip_data(self, sample_data, sample_ip_data):
        """Test IP data merging."""
        preprocessor = FraudDataPreprocessor()
        
        # Clean both datasets
        cleaned_fraud = preprocessor.clean_data(sample_data)
        cleaned_ip = sample_ip_data.copy()
        
        # Merge data
        merged_data = preprocessor.merge_with_ip_data(cleaned_fraud, cleaned_ip)
        
        # Check country column exists
        assert 'country' in merged_data.columns
        
        # Check all rows have country info
        assert merged_data['country'].notnull().all()
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical feature encoding."""
        preprocessor = FraudDataPreprocessor()
        
        # Clean data
        cleaned_data = preprocessor.clean_data(sample_data)
        
        # Encode categorical features
        categorical_cols = ['source', 'browser', 'sex']
        encoded_data = preprocessor.encode_categorical_features(cleaned_data, categorical_cols)
        
        # Check encoded columns exist
        for col in categorical_cols:
            encoded_col = f'{col}_encoded'
            assert encoded_col in encoded_data.columns
            
            # Check encoding is numeric
            assert pd.api.types.is_numeric_dtype(encoded_data[encoded_col])
    
    def test_scale_numerical_features(self, sample_data):
        """Test numerical feature scaling."""
        preprocessor = FraudDataPreprocessor()
        
        # Clean data
        cleaned_data = preprocessor.clean_data(sample_data)
        
        # Scale numerical features
        numerical_cols = ['purchase_value', 'age']
        scaled_data = preprocessor.scale_numerical_features(
            cleaned_data, numerical_cols, scaler_type='standard'
        )
        
        # Check scaled columns exist
        for col in numerical_cols:
            scaled_col = f'{col}_scaled'
            assert scaled_col in scaled_data.columns
            
            # Check scaling (mean should be close to 0 for standard scaling)
            if scaled_col in scaled_data.columns:
                assert abs(scaled_data[scaled_col].mean()) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])