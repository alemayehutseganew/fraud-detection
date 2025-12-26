"""Tests for Streamlit helper functions that interact with the backend API (app.py).

Specifically tests `call_api_predict` by mocking `requests.post` to simulate API responses and errors.
"""
import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from app import call_api_predict


class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def test_call_api_predict_success(monkeypatch):
    expected = {"predictions": [1], "probabilities": [0.85], "n": 1}

    def fake_post(url, json, timeout):
        assert 'instance' in json or 'instances' in json
        return DummyResponse(expected, status_code=200)

    monkeypatch.setattr('requests.post', fake_post)

    features = {"purchase_value": 100}
    resp = call_api_predict("http://fake/api/predict", features)
    assert resp == expected


def test_call_api_predict_error(monkeypatch):
    def fake_post_fail(url, json, timeout):
        return DummyResponse({"error": "bad"}, status_code=500)

    monkeypatch.setattr('requests.post', fake_post_fail)

    with pytest.raises(Exception):
        call_api_predict("http://fake/api/predict", {"purchase_value": 1})
