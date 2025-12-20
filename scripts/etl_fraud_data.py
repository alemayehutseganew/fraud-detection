"""ETL helpers for Fraud_Data.csv

Creates cleaned and feature-engineered output in data/processed/

Usage:
    python scripts/etl_fraud_data.py

The script expects the following files in `data/raw/`:
- Fraud_Data.csv
- IpAddress_to_Country.csv

Outputs:
- data/processed/fraud_data_processed.csv
"""
from pathlib import Path
import pandas as pd
import ipaddress
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
# Use the `fraud-detection` package root: data is under this folder
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def ip_to_int(ip: str) -> int:
    try:
        return int(ipaddress.IPv4Address(ip))
    except Exception:
        return pd.NA


def load_fraud_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        logging.error(f"Missing fraud data at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logging.info(f"Loaded fraud data with {len(df):,} rows")
    return df


def load_ip_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        logging.error(f"Missing IP map at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logging.info(f"Loaded IP map with {len(df):,} rows")
    return df


def preprocess_fraud(df: pd.DataFrame, ip_map: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    logging.info(f"Dropped duplicates: {before - len(df):,}")

    # Parse timestamps
    for col in ["signup_time", "purchase_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Basic missing-value handling: report and drop rows missing target or purchase_time
    if "class" in df.columns:
        missing_target = df["class"].isna().sum()
        logging.info(f"Missing target values: {missing_target}")
    missing_purchase = df["purchase_time"].isna().sum() if "purchase_time" in df.columns else 0
    logging.info(f"Missing purchase_time: {missing_purchase}")

    df = df.dropna(subset=[c for c in ["class", "purchase_time"] if c in df.columns])

    # Numeric conversion
    if "purchase_value" in df.columns:
        df["purchase_value"] = pd.to_numeric(df["purchase_value"], errors="coerce")

    # Feature engineering: hour_of_day, day_of_week, time_since_signup
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    if "signup_time" in df.columns:
        df["time_since_signup_seconds"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    else:
        df["time_since_signup_seconds"] = pd.NA

    # Convert IP to integer
    if "ip_address" in df.columns:
        df["ip_int"] = df["ip_address"].astype(str).apply(ip_to_int)
    else:
        df["ip_int"] = pd.NA

    # Prepare IP map and merge via range lookup
    if not ip_map.empty and {"lower_bound_ip_address", "upper_bound_ip_address", "country"}.issubset(ip_map.columns):
        ip_map = ip_map.drop_duplicates().copy()
        ip_map["lower_int"] = pd.to_numeric(ip_map["lower_bound_ip_address"].astype(str).apply(lambda x: ip_to_int(x)), errors="coerce")
        ip_map["upper_int"] = pd.to_numeric(ip_map["upper_bound_ip_address"].astype(str).apply(lambda x: ip_to_int(x)), errors="coerce")
        ip_map = ip_map.dropna(subset=["lower_int"]).sort_values("lower_int")

        # Ensure numeric ip_int for merging
        df = df.copy()
        df["ip_int"] = pd.to_numeric(df["ip_int"], errors="coerce")

        # Merge asof on ip_int >= lower_int, then filter where ip_int <= upper_int
        df_sorted = df.dropna(subset=["ip_int"]).sort_values("ip_int").copy()
        merged = pd.merge_asof(df_sorted, ip_map, left_on="ip_int", right_on="lower_int", direction="backward")
        # Keep only matches where ip_int <= upper_int
        merged["country"] = merged.apply(lambda r: r["country"] if pd.notna(r.get("upper_int")) and r["ip_int"] <= r["upper_int"] else pd.NA, axis=1)

        # Combine back country into original df
        df = df.merge(merged[["user_id", "country"]], on="user_id", how="left")
    else:
        df["country"] = pd.NA

    # Fill simple categorical NaNs
    for c in ["source", "browser", "sex", "country"]:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")

    return df


def main():
    fraud_path = RAW / "Fraud_Data.csv"
    ip_map_path = RAW / "IpAddress_to_Country.csv"
    out_path = PROCESSED / "fraud_data_processed.csv"

    fraud = load_fraud_data(fraud_path)
    ip_map = load_ip_map(ip_map_path)

    if fraud.empty:
        logging.error("No fraud data to process. Place `Fraud_Data.csv` in data/raw/ and rerun.")
        return

    processed = preprocess_fraud(fraud, ip_map)
    processed.to_csv(out_path, index=False)
    logging.info(f"Wrote processed data to {out_path} ({len(processed):,} rows)")


if __name__ == "__main__":
    main()
