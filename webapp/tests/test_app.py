import importlib.util
import os
from pathlib import Path


def load_app_module():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("careagent_test_app", app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_health_and_device_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv("DEVICE_API_KEY", "test-key")
    module = load_app_module()
    app = module.create_app({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
    })
    payload = {
        "event_id": "test-event-1", "deviceId": "rpi-01", "captured_at": "2026-06-16T20:00:00+00:00",
        "dht22": {"gpio_pin": 4, "timestamp": "2026-06-16T20:00:00+00:00", "temperature_celsius": 23, "humidity_percent": 50},
        "PIR501": {"value": 1, "gpio_pin": 3, "timestamp": "2026-06-16T20:00:00+00:00"},
        "pose": "STANDING", "pose_confidence": 0.8, "health_status": "Normal", "emergency": False,
        "stroke_risk": {"risk_level": "LOW", "facial_droop": False, "arm_weakness": False, "sudden_collapse": False, "pose_freeze": False, "gradual_deterioration": False, "signs_count": 0},
    }
    with app.test_client() as client:
        assert client.get("/health").status_code == 200
        first = client.post("/api/readings", json=payload, headers={"X-API-Key": "test-key"})
        assert first.status_code == 201
        duplicate = client.post("/api/readings", json=payload, headers={"X-API-Key": "test-key"})
        assert duplicate.status_code == 200
        assert duplicate.get_json()["status"] == "duplicate"


def test_login_and_responsive_dashboard(tmp_path, monkeypatch):
    import re
    monkeypatch.setenv("DEVICE_API_KEY", "test-key")
    module = load_app_module()
    app = module.create_app({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'login.db'}",
    })
    with app.test_client() as client:
        login_page = client.get("/login")
        assert login_page.status_code == 200
        token = re.search(rb'name="csrf_token" value="([^"]+)"', login_page.data).group(1).decode()
        response = client.post("/login", data={"username": "admin", "password": "Admin@123", "csrf_token": token}, follow_redirects=True)
        assert response.status_code == 200
        assert b"Patient Safety Dashboard" in response.data
        assert b"mobile-header" in response.data
        assert b"Emergency action" in response.data
