# Fraud Detection for E-commerce and Bank Transactions

## Project Overview
This project aims to improve fraud detection using machine learning models trained on e-commerce and bank transaction data for Adey Innovations Inc.

## Business Context
Adey Innovations Inc. needs accurate fraud detection models that balance:
- **Security** (catching fraud)
- **User experience** (minimizing false positives)
- **Real-time monitoring** capabilities

## Dataset Description

## Preprocessing & Engineered Features
- Missing values are handled (numeric -> median imputation; categorical -> mode or 'unknown' for important fields). Duplicates are removed and timestamps are parsed.
- **Geolocation:** IP addresses are converted to integer form and merged with the IP range-to-country map to enable country-level analysis.
- **Time-based features:** hour_of_day, day_of_week, month, time_since_signup, same_day_purchase.
- **Transaction frequency & velocity:** per-user transaction counts in rolling time windows (24h, 7d, 30d) and average time-between-transactions to capture velocity.
- **Categorical encoding:** `LabelEncoder` (ordinal) is available, and `one_hot_encode` is implemented for models that require one-hot inputs.
- **Scaling:** Numerical features can be scaled with `StandardScaler` or `MinMaxScaler`.
- **Class imbalance:** SMOTE is applied to training folds only (synthetic minority examples) to improve decision boundaries while avoiding data leakage into validation/test sets.


### 1. Fraud_Data.csv (E-commerce)
- **user_id**: Unique user identifier
- **signup_time**: Timestamp when user signed up
- **purchase_time**: Transaction timestamp
- **purchase_value**: Transaction amount in USD
- **device_id**: Device identifier
- **source**: Traffic source (SEO, Ads, etc.)
- **browser**: Browser used
- **sex**: User gender (M/F)
- **age**: User age
- **ip_address**: Transaction IP address
- **class**: Target (1 = Fraud, 0 = Legitimate)

### 2. IpAddress_to_Country.csv
- Maps IP address ranges to countries
- Used for geolocation analysis

### 3. creditcard.csv (Bank Transactions)
- **Time**: Seconds elapsed since first transaction
- **V1-V28**: PCA-transformed features (anonymized)
- **Amount**: Transaction amount
- **Class**: Target (1 = Fraud, 0 = Legitimate)

## Project Structure