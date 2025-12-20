# Fraud Detection for E-commerce and Bank Transactions

## Project Overview
This project aims to improve fraud detection using machine learning models trained on e-commerce and bank transaction data for Adey Innovations Inc.

## Business Context
Adey Innovations Inc. needs accurate fraud detection models that balance:
- **Security** (catching fraud)
- **User experience** (minimizing false positives)
- **Real-time monitoring** capabilities

## Dataset Description

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