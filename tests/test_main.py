import pytest
from fastapi.testclient import TestClient
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Hospital Readmission Risk Scorer API" in response.json()["name"]

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "XGBoost Classifier" in response.json()["model_type"]
