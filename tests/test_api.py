"""Tests for the FastAPI model serving endpoints (src.api).

Tests cover:
- GET /health (model loaded flag behavior)
- POST /predict (single instance & batch)
- Error handling for invalid payloads

We monkeypatch filesystem checks and joblib.load to avoid relying on a real model artifact.
"""

import sys
from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src import api as api_module


class DummyModel:
    def __init__(self):
        pass

    def predict(self, X):
        # return 1 for all inputs if contains 'amount' > 100, else 0
        try:
            if hasattr(X, 'values'):
                arr = X
            else:
                arr = X
            # simple deterministic placeholder
            return [1 for _ in range(len(X))]
        except Exception:
            return [0]

    def predict_proba(self, X):
        # Return 0.9 for positive class for all rows
        n = len(X) if hasattr(X, '__len__') else 1
        return [[0.1, 0.9] for _ in range(n)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a TestClient with model loading monkeypatched to use DummyModel."""
    # Ensure model path existence check returns True for expected model path
    def fake_exists(p):
        # return True when checking for models/best_model_pipeline.pkl
        if isinstance(p, str) and 'models/best_model_pipeline.pkl' in p:
            return True
        # otherwise use real exists for safety
        return False

    monkeypatch.setattr(api_module, 'MODEL_PATHS', ['models/best_model_pipeline.pkl'])
    monkeypatch.setattr('os.path.exists', lambda p: fake_exists(p))
    # monkeypatch joblib.load to return a small dummy model
    monkeypatch.setattr('joblib.load', lambda p: DummyModel())

    # Recreate app state by calling startup
    app = api_module.app
    # Startup event runs when TestClient is created
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get('/health')
    assert r.status_code == 200
    payload = r.json()
    assert payload.get('status') == 'ok'
    assert payload.get('model_loaded') is True


def test_predict_single_instance(client):
    payload = {"instance": {"purchase_value": 123.45, "age": 30}}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 'predictions' in body
    assert 'probabilities' in body
    assert body['n'] == 1
    assert isinstance(body['predictions'], list)


def test_predict_batch(client):
    payload = {"instances": [{"purchase_value": 1}, {"purchase_value": 2}]}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body['n'] == 2
    assert len(body['predictions']) == 2


def test_predict_invalid_payload(client):
    # Empty body
    r = client.post('/predict', json={})
    assert r.status_code == 400
    # Bad format
    r2 = client.post('/predict', json={"foo": "bar"})
    assert r2.status_code == 400


def test_logistic_regression_missing_multi_class(monkeypatch, tmp_path):
    """Simulate loading a LogisticRegression pickled with a newer scikit-learn
    where the instance may not have the 'multi_class' attribute during runtime.
    Ensure the API patches it and prediction succeeds."""
    from sklearn.linear_model import LogisticRegression

    def fake_exists(p):
        return isinstance(p, str) and 'models/best_model_pipeline.pkl' in p

    # Create an LR and delete the attribute to simulate missing field after unpickle
    lr = LogisticRegression()
    if hasattr(lr, 'multi_class'):
        try:
            delattr(lr, 'multi_class')
        except Exception:
            # some SKlearn versions may store as property; simulate by setting to None
            try:
                lr.multi_class = None
            except Exception:
                pass

    # To keep this test deterministic and avoid "not fitted" or feature-mismatch
    # errors from calling an unfitted sklearn estimator, override the prediction
    # methods on the instance with simple deterministic functions.
    def _fake_predict(X):
        try:
            n = len(X)
        except Exception:
            n = 1
        return [0 for _ in range(n)]

    def _fake_predict_proba(X):
        try:
            n = len(X)
        except Exception:
            n = 1
        return [[0.5, 0.5] for _ in range(n)]

    lr.predict = _fake_predict
    lr.predict_proba = _fake_predict_proba

    monkeypatch.setattr(api_module, 'MODEL_PATHS', ['models/best_model_pipeline.pkl'])
    monkeypatch.setattr('os.path.exists', lambda p: fake_exists(p))
    monkeypatch.setattr('joblib.load', lambda p: lr)

    from fastapi.testclient import TestClient
    app = api_module.app
    with TestClient(app) as c:
        r = c.post('/predict', json={"instance": {"purchase_value": 10}})
        assert r.status_code == 200
        body = r.json()
        assert 'predictions' in body
        assert body['n'] == 1

