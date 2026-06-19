from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmergencyDecision:
    emergency: bool
    reasons: list[str]
    environment_status: str


class EmergencyEngine:
    """Separates room-environment warnings from medical/emergency alerts."""

    @staticmethod
    def evaluate(payload: dict[str, Any], manual_button: bool = False) -> EmergencyDecision:
        risk = payload.get("stroke_risk", {}) if isinstance(payload.get("stroke_risk"), dict) else {}
        pose = str(payload.get("pose", "UNKNOWN")).upper()
        pose_confidence = float(payload.get("pose_confidence", 0.0) or 0.0)
        pir_motion = int(payload.get("PIR501", {}).get("value", 0) or 0)
        temperature = float(payload.get("dht22", {}).get("temperature_celsius", 0.0) or 0.0)
        humidity = float(payload.get("dht22", {}).get("humidity_percent", 0.0) or 0.0)

        environment_warnings = []
        if temperature and not 16 <= temperature <= 30:
            environment_warnings.append("room temperature outside configured comfort range")
        if humidity and not 25 <= humidity <= 75:
            environment_warnings.append("room humidity outside configured comfort range")
        environment_status = "WARNING: " + "; ".join(environment_warnings) if environment_warnings else "NORMAL"

        reasons: list[str] = []
        if manual_button:
            reasons.append("manual emergency button")
        if str(risk.get("risk_level", "LOW")).upper() == "CRITICAL":
            reasons.append("critical camera-based risk pattern")
        if bool(risk.get("sudden_collapse")) and bool(risk.get("pose_freeze")):
            reasons.append("collapse followed by immobility")
        if pose in {"FALLING", "COLLAPSING"} and pose_confidence >= 0.82 and pir_motion == 0:
            reasons.append("high-confidence collapse pose with no PIR motion")

        return EmergencyDecision(emergency=bool(reasons), reasons=reasons, environment_status=environment_status)
