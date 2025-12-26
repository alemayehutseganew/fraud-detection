import requests
import json

url = 'http://127.0.0.1:8000/predict'
payload = {'instance': {'purchase_value': 123.45}}
resp = requests.post(url, json=payload, timeout=10)
print('status', resp.status_code)
try:
    print(resp.json())
except Exception:
    print(resp.text)
