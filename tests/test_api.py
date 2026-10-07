from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "har_d": 0.00003,
    "har_w": 0.00004,
    "har_m": 0.00005,
    "rv_lag1": 0.00003,
    "abs_return": 0.01,
    "hl_range": 0.016,
    "log_volume": 18.0,
}


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_non_negative_variance():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()

    assert body["predicted_variance"] >= 0
    assert "monitoring" in body
    assert body["monitoring"]["warning_count"] == 0


def test_predict_rejects_negative_variance_input():
    invalid_payload = {**VALID_PAYLOAD, "har_d": -0.00001}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_monitor_reports_outside_training_reference_range():
    outside_range_payload = {**VALID_PAYLOAD, "har_d": 0.0005}

    response = client.post("/monitor", json=outside_range_payload)

    assert response.status_code == 200
    body = response.json()

    assert body["warning_count"] >= 1
    assert any(
        warning["feature"] == "har_d"
        for warning in body["warnings"]
    )
