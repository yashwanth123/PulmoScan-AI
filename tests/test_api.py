"""FastAPI integration tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "PulmoScan AI"


def test_model_info(client) -> None:
    response = client.get("/api/model")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert "architecture" in data
    assert data["input_size"] == [224, 224]


def test_scan_types(client) -> None:
    response = client.get("/api/scan-types")
    assert response.status_code == 200
    types = response.json()["scan_types"]
    assert any(item["key"] == "chest_xray" for item in types)


def test_stats(client) -> None:
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_scans" in data
    assert "by_class" in data


def test_history(client) -> None:
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "history" in response.json()


def test_frontend_index(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PulmoScan" in response.text


def test_samples_list(client) -> None:
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "chest_xray" in data["samples"]
    assert "ct_scan" in data["samples"]
    assert len(data["samples"]["ct_scan"]) >= 1


def test_predict_rejects_empty_file(client) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("empty.png", b"", "image/png")},
        data={"scan_type": "chest_xray"},
    )
    assert response.status_code == 400


def test_predict_rejects_invalid_extension(client) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("scan.txt", b"not an image", "text/plain")},
        data={"scan_type": "chest_xray"},
    )
    assert response.status_code == 400


@pytest.mark.slow
def test_predict_with_image(client, sample_xray_bytes: bytes) -> None:
    response = client.post(
        "/api/predict",
        files={"file": ("xray.png", sample_xray_bytes, "image/png")},
        data={"scan_type": "chest_xray", "include_gradcam": "false"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "diagnosis" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "risk_level" in data

    stats = client.get("/api/stats").json()
    assert stats["total_scans"] >= 1
