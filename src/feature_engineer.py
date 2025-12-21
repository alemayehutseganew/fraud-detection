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
            # Transaction frequency & velocity features
            df_features = self.create_transaction_features(df_features)
        
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

    def create_transaction_features(self, df, windows_hours=(24, 168, 720)):
        """Create transaction frequency and velocity features per user.

        windows_hours: tuple of window sizes in hours (e.g., 24, 168 (7d), 720 (30d))
        Adds columns: txn_count_{h}h for each window and avg_time_between_txn_hours
        """
        df_tx = df.copy()
        if 'purchase_time' not in df_tx.columns or 'user_id' not in df_tx.columns:
            return df_tx

        # Ensure datetime and sort
        df_tx = df_tx.dropna(subset=['purchase_time']).copy()
        df_tx['purchase_time'] = pd.to_datetime(df_tx['purchase_time'])
        df_tx = df_tx.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)

        # Convert to integer ns for fast arithmetic
        times_ns = df_tx['purchase_time'].astype('int64').values
        user_ids = df_tx['user_id'].values

        # Prepare containers
        for h in windows_hours:
            col = f'txn_count_{h}h'
            df_tx[col] = 0

        df_tx['time_diff_hours'] = np.nan

        # Process per user
        unique_users = pd.unique(user_ids)
        for user in unique_users:
            mask = (user_ids == user)
            idxs = np.nonzero(mask)[0]
            user_times = times_ns[idxs]
            if len(user_times) == 0:
                continue

            # compute time diffs (hours) between consecutive txns
            diffs = np.diff(user_times) / 1e9 / 3600.0
            # Time diff for first txn is NaN
            user_time_diff = np.concatenate(([np.nan], diffs))
            df_tx.loc[df_tx.index[idxs], 'time_diff_hours'] = user_time_diff

            # Sliding counts using searchsorted
            for h in windows_hours:
                window_ns = int(h * 3600 * 1e9)
                counts = np.empty(len(user_times), dtype=int)
                for i, t in enumerate(user_times):
                    left = np.searchsorted(user_times, t - window_ns, side='left')
                    counts[i] = i - left + 1
                df_tx.loc[df_tx.index[idxs], f'txn_count_{h}h'] = counts

        # Aggregate velocity: average of recent diffs (previous 3 diffs)
        df_tx['avg_time_between_txn_hours'] = (
            df_tx['time_diff_hours'].rolling(window=3, min_periods=1).mean()
        )

        # Some rows may no longer align with original index if original df had different ordering
        # Merge back into original DataFrame
        merge_cols = ['user_id', 'purchase_time', 'txn_count_24h', 'txn_count_168h', 'txn_count_720h', 'avg_time_between_txn_hours']
        merge_available = [c for c in merge_cols if c in df_tx.columns]
        df_out = df.merge(df_tx[merge_available], on=['user_id', 'purchase_time'], how='left')

        # Fill NaNs with 0 for counts
        for h in windows_hours:
            col = f'txn_count_{h}h'
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(0).astype(int)

        if 'avg_time_between_txn_hours' in df_out.columns:
            df_out['avg_time_between_txn_hours'] = df_out['avg_time_between_txn_hours'].fillna(pd.NA)

        print("Added transaction frequency and velocity features")
        return df_out