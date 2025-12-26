import requests
r = requests.get('http://127.0.0.1:8000/health', timeout=5)
print(r.status_code)
print(r.json())
