"""Example Streamlit front-end that uses the FastAPI model endpoint (/predict).
Run the FastAPI service first: uvicorn src.api:app --reload --port 8000
Then run this with: streamlit run streamlit_client_example.py

This is a minimal example — adapt the form fields to match your model's feature set.
"""
import streamlit as st
import requests

API_URL = st.text_input("API URL", "http://localhost:8000/predict")

st.title("Fraud Detection — Streamlit Client (Example)")

# For demo, allow users to paste JSON or use a simple form
mode = st.radio("Input mode", ["Form", "JSON"])

if mode == "Form":
    st.write("Fill a sample record (customize fields to match training features)")
    # Example fields — replace with actual columns from your dataset
    amount = st.number_input("amount", min_value=0.0, value=100.0)
    device_id = st.text_input("device_id", value="device_1")
    ip_address = st.text_input("ip_address", value="127.0.0.1")

    if st.button("Predict"):
        payload = {"instance": {"amount": amount, "device_id": device_id, "ip_address": ip_address}}
        try:
            resp = requests.post(API_URL, json=payload)
            resp.raise_for_status()
            st.json(resp.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

else:
    raw = st.text_area("JSON input (single instance or {\"instances\": [...]})")
    if st.button("Send JSON"):
        try:
            import json
            j = json.loads(raw)
            resp = requests.post(API_URL, json=j)
            resp.raise_for_status()
            st.json(resp.json())
        except Exception as e:
            st.error(f"Request failed or invalid JSON: {e}")
