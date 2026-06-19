from __future__ import annotations

from typing import Any


def deterministic_health_status(payload: dict[str, Any]) -> str:
    if payload.get("emergency"):
        return "Emergency attention required"
    risk = str(payload.get("stroke_risk", {}).get("risk_level", "UNKNOWN")).upper()
    if risk == "HIGH":
        return "Check patient now"
    if risk == "MEDIUM":
        return "Observation recommended"
    if payload.get("pose") == "UNKNOWN":
        return "Person not reliably detected"
    if str(payload.get("environment_status", "NORMAL")).startswith("WARNING"):
        return "Environment warning"
    return "Normal"
