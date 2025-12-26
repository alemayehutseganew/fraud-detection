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

## Run the Streamlit demo

To quickly run the interactive demo locally:

Windows (PowerShell):

1. Create a virtual environment and install dependencies:

```powershell
cd fraud-detection
scripts\setup_streamlit_env.ps1
```

2. Activate the environment and run the app:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Or install dependencies manually:

```powershell
pip install -r requirements-streamlit.txt
streamlit run app.py
```

Deploy to Streamlit Cloud:
- Push the repo with `app.py` and `requirements-streamlit.txt` to GitHub.
- Add the repo to Streamlit Community Cloud and set the entrypoint to `app.py`.

FastAPI backend (optional):

If you want to separate the model serving from the Streamlit UI, run the FastAPI model service and point the Streamlit frontend to it.

1. Install FastAPI and Uvicorn:

```powershell
pip install fastapi uvicorn[standard]
```

2. Run the API (from repo root):

```powershell
uvicorn src.api:app --reload --port 8000
```

3. In your Streamlit app (or the provided `streamlit_client_example.py`), set the API URL to `http://localhost:8000/predict`.

The FastAPI service will try to load the saved pipeline at `models/best_model_pipeline.pkl` and expose a POST `/predict` endpoint that accepts `{"instance": {...}}` or `{"instances": [{...}, ...]}`. If no model is found the `/predict` endpoint will return a helpful error telling you to save the model artifact.

---

## Docker deployment (optional) 🐳

You can containerize the Streamlit app using the included `Dockerfile`.

1) Build the image (from the repo root):

```bash
docker build -t myorg/fraud-detection:latest .
```

2) Run the container locally and map the port:

```bash
docker run -p 8501:8501 myorg/fraud-detection:latest
```

3) Push to a registry (Docker Hub / GitHub Container Registry):

```bash
# Tag (if needed) and push
docker tag myorg/fraud-detection:latest <YOUR_REGISTRY>/fraud-detection:latest
docker push <YOUR_REGISTRY>/fraud-detection:latest
```

Notes & tips:
- The Docker image installs the exact versions from `requirements-streamlit.txt`. If you'd like a smaller image, we can use a multi-stage build or switch to `python:3.12-alpine` (may need extra packages).
- Ensure large data files (in `data/`) are not included in the image — `.dockerignore` excludes `data/` by default.
- To run in production behind a reverse proxy or in Kubernetes, set the container to expose port `8501` and use the same startup command as shown in the `Dockerfile`.

## Docker Compose (local development)

`docker-compose.yml` is provided to make local development and testing easy. It mounts your repo into the container so code changes are picked up without rebuilding the image.

1) Build and start (foreground):

```bash
docker compose up --build
```

2) Run in detached mode:

```bash
docker compose up -d --build
```

3) Stop and remove containers:

```bash
docker compose down
```

Notes:
- The service `web` maps the container port `8501` to your host, so the app will be reachable at `http://localhost:8501`.
- If you make changes to dependencies, rebuild with `docker compose up --build` to apply them.


