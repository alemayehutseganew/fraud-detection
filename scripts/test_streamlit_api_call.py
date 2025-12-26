import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from app import call_api_predict

features = {'purchase_value': 120}
print(call_api_predict('http://127.0.0.1:8002/predict', features))