from emergency_engine import EmergencyEngine


def payload():
    return {
        "pose": "STANDING", "pose_confidence": 0.9,
        "PIR501": {"value": 1},
        "dht22": {"temperature_celsius": 23, "humidity_percent": 50},
        "stroke_risk": {"risk_level": "LOW", "sudden_collapse": False, "pose_freeze": False},
    }


def test_environment_warning_is_not_emergency():
    data = payload()
    data["dht22"]["temperature_celsius"] = 10
    decision = EmergencyEngine.evaluate(data)
    assert decision.emergency is False
    assert decision.environment_status.startswith("WARNING")


def test_collapse_and_freeze_is_emergency():
    data = payload()
    data["stroke_risk"].update({"risk_level": "CRITICAL", "sudden_collapse": True, "pose_freeze": True})
    decision = EmergencyEngine.evaluate(data)
    assert decision.emergency is True
    assert "collapse followed by immobility" in decision.reasons
