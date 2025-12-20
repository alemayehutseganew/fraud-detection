"""
Data preprocessing functions for fraud detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FraudDataPreprocessor:
    """Preprocess e-commerce fraud data."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scalers = {}
        self.encoders = {}
        
    def clean_data(self, df):
        """Clean the fraud dataset."""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Convert timestamps
        if 'signup_time' in df_clean.columns:
            df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
        if 'purchase_time' in df_clean.columns:
            df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"Filled missing values in {col} with median")
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                print(f"Filled missing values in {col} with mode")
        
        return df_clean
    
    def convert_ip_to_int(self, ip_address):
        """Convert IP address to integer for range lookup."""
        if pd.isnull(ip_address):
            return None
        if isinstance(ip_address, (int, float)):
            return int(ip_address)
        try:
            parts = str(ip_address).split('.')
            if len(parts) == 4:
                return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
            else:
                return None
        except:
            return None
    
    def merge_with_ip_data(self, fraud_df, ip_df):
        """Merge fraud data with IP country data using range lookup."""
        fraud_df_clean = fraud_df.copy()
        ip_df_clean = ip_df.copy()
        
        # Convert IP addresses to integers
        fraud_df_clean['ip_int'] = fraud_df_clean['ip_address'].apply(self.convert_ip_to_int)
        ip_df_clean['lower_int'] = ip_df_clean['lower_bound_ip_address'].apply(self.convert_ip_to_int)
        ip_df_clean['upper_int'] = ip_df_clean['upper_bound_ip_address'].apply(self.convert_ip_to_int)
        
        # Drop rows with NA
        fraud_df_clean = fraud_df_clean.dropna(subset=['ip_int'])
        ip_df_clean = ip_df_clean.dropna(subset=['lower_int', 'upper_int'])
        
        fraud_df_clean['ip_int'] = pd.to_numeric(fraud_df_clean['ip_int'], errors='coerce')
        ip_df_clean['lower_int'] = pd.to_numeric(ip_df_clean['lower_int'], errors='coerce')
        ip_df_clean['upper_int'] = pd.to_numeric(ip_df_clean['upper_int'], errors='coerce')
        
        # Drop any new NaN
        fraud_df_clean = fraud_df_clean.dropna(subset=['ip_int'])
        ip_df_clean = ip_df_clean.dropna(subset=['lower_int', 'upper_int'])
        
        print("After second dropna: fraud_df_clean shape:", fraud_df_clean.shape)
        print("ip_df_clean shape:", ip_df_clean.shape)
        
        fraud_df_clean['ip_int'] = pd.to_numeric(fraud_df_clean['ip_int'], errors='coerce', downcast='integer')
        ip_df_clean['lower_int'] = pd.to_numeric(ip_df_clean['lower_int'], errors='coerce', downcast='integer')
        ip_df_clean['upper_int'] = pd.to_numeric(ip_df_clean['upper_int'], errors='coerce', downcast='integer')
        
        # Drop any NaN from to_numeric
        fraud_df_clean = fraud_df_clean.dropna(subset=['ip_int'])
        ip_df_clean = ip_df_clean.dropna(subset=['lower_int', 'upper_int'])
        
        print("After to_numeric: fraud_df_clean['ip_int'] dtype:", fraud_df_clean['ip_int'].dtype)
        print("ip_df_clean['lower_int'] dtype:", ip_df_clean['lower_int'].dtype)
        
        # Sort for merge_asof
        ip_df_sorted = ip_df_clean.sort_values('lower_int')
        fraud_df_sorted = fraud_df_clean.sort_values('ip_int')
        
        print("Before merge:")
        print("fraud_df_sorted['ip_int'] dtype:", fraud_df_sorted['ip_int'].dtype)
        print("ip_df_sorted['lower_int'] dtype:", ip_df_sorted['lower_int'].dtype)
        print("ip_df_sorted['upper_int'] dtype:", ip_df_sorted['upper_int'].dtype)
        
        # Merge using merge_asof
        merged_df = pd.merge_asof(
            fraud_df_sorted,
            ip_df_sorted[['lower_int', 'upper_int', 'country']],
            left_on='ip_int',
            right_on='lower_int',
            direction='backward'
        )
        
        # Filter valid matches
        merged_df = merged_df[
            (merged_df['ip_int'] >= merged_df['lower_int']) & 
            (merged_df['ip_int'] <= merged_df['upper_int'])
        ]
        
        print(f"Merged {len(merged_df):,} rows with country information")
        
        return merged_df
    
    def create_time_features(self, df):
        """Create time-based features."""
        df_features = df.copy()
        
        if 'purchase_time' in df_features.columns:
            # Hour and day features
            df_features['purchase_hour'] = df_features['purchase_time'].dt.hour
            df_features['purchase_day'] = df_features['purchase_time'].dt.day
            df_features['purchase_dayofweek'] = df_features['purchase_time'].dt.dayofweek
            df_features['purchase_month'] = df_features['purchase_time'].dt.month
            
            # Time since signup
            if 'signup_time' in df_features.columns:
                df_features['time_since_signup'] = (
                    df_features['purchase_time'] - df_features['signup_time']
                ).dt.total_seconds() / 3600  # Convert to hours
                
                # Same day purchase flag
                df_features['same_day_purchase'] = (
                    df_features['purchase_time'].dt.date == df_features['signup_time'].dt.date
                ).astype(int)
        
        return df_features
    
    def encode_categorical_features(self, df, categorical_cols):
        """Encode categorical features."""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                print(f"Encoded {col} with {len(le.classes_)} categories")
        
        return df_encoded
    
    def scale_numerical_features(self, df, numerical_cols, scaler_type='standard'):
        """Scale numerical features."""
        df_scaled = df.copy()
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Store scaler for later use
        scaler_key = f"{scaler_type}_scaler"
        self.scalers[scaler_key] = scaler
        
        # Scale features
        for col in numerical_cols:
            if col in df_scaled.columns:
                df_scaled[f'{col}_scaled'] = scaler.fit_transform(df_scaled[[col]])
                print(f"Scaled {col} using {scaler_type} scaler")
        
        return df_scaled


class CreditCardPreprocessor:
    """Preprocess credit card fraud data."""
    
    def __init__(self):
        """Initialize credit card preprocessor."""
        self.scaler = StandardScaler()
        
    def clean_data(self, df):
        """Clean credit card data."""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Scale 'Amount' feature
        df_clean['Amount_scaled'] = self.scaler.fit_transform(df_clean[['Amount']])
        
        # Time-based features
        df_clean['Time_hour'] = (df_clean['Time'] // 3600) % 24
        df_clean['Time_day'] = (df_clean['Time'] // (3600 * 24)) % 7
        
        return df_clean