import requests
print('health ->', requests.get('http://127.0.0.1:8002/health').json())
print('predict ->', requests.post('http://127.0.0.1:8002/predict', json={'instance': {'purchase_value': 120}}).json())