from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass
from typing import Any

from pose_features import distance, midpoint, visible


@dataclass
class SignalState:
    facial_droop: bool = False
    arm_weakness: bool = False
    sudden_collapse: bool = False
    pose_freeze: bool = False
    gradual_deterioration: bool = False


class StrokeRiskEngine:
    """Conservative camera-based warning engine.

    This does not diagnose stroke. It detects persistent visual asymmetry and motion
    patterns that can justify checking the person. Face and arm indicators are
    down-weighted unless landmarks are visible and the body is approximately frontal.
    """

    def __init__(self, visibility_threshold: float = 0.6, history_seconds: float = 12.0):
        self.visibility_threshold = visibility_threshold
        self.history_seconds = history_seconds
        self.history: collections.deque[dict[str, Any]] = collections.deque(maxlen=180)
        self.boolean_history: collections.deque[SignalState] = collections.deque(maxlen=15)
        self.last_collapse_timestamp_ms: int | None = None

    @staticmethod
    def _landmarks(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
        landmarks = observation.get("landmarks", observation)
        return landmarks if isinstance(landmarks, dict) else {}

    def _prune(self, newest_timestamp_ms: int) -> None:
        cutoff = newest_timestamp_ms - int(self.history_seconds * 1000)
        while self.history and int(self.history[0].get("timestamp_ms", 0)) < cutoff:
            self.history.popleft()

    def _frontal_face(self, lm: dict[str, dict[str, Any]]) -> bool:
        needed = ["LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in needed):
            return False
        shoulder_depth_diff = abs(float(lm["LEFT_SHOULDER"].get("z", 0)) - float(lm["RIGHT_SHOULDER"].get("z", 0)))
        eye_width = distance(lm["LEFT_EYE"], lm["RIGHT_EYE"], include_z=False)
        ear_width = distance(lm["LEFT_EAR"], lm["RIGHT_EAR"], include_z=False)
        return eye_width > 0.018 and ear_width > eye_width * 1.15 and shoulder_depth_diff < 0.22

    def _face_asymmetry_score(self, lm: dict[str, dict[str, Any]]) -> float:
        names = ["MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_EYE", "RIGHT_EYE"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in names) or not self._frontal_face(lm):
            return 0.0
        mouth_diff = abs(float(lm["MOUTH_LEFT"]["y"]) - float(lm["MOUTH_RIGHT"]["y"]))
        eye_diff = abs(float(lm["LEFT_EYE"]["y"]) - float(lm["RIGHT_EYE"]["y"]))
        face_scale = max(distance(lm["LEFT_EYE"], lm["RIGHT_EYE"], include_z=False), 0.015)
        # Pose Landmarker has sparse face landmarks, so require a strong normalized signal.
        normalized = (mouth_diff + 0.5 * eye_diff) / face_scale
        return min(1.0, normalized / 0.34)

    def _arm_asymmetry_score(self, lm: dict[str, dict[str, Any]]) -> float:
        names = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in names):
            return 0.0
        shoulder_width = max(distance(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"], include_z=False), 0.03)
        wrist_diff = abs(float(lm["LEFT_WRIST"]["y"]) - float(lm["RIGHT_WRIST"]["y"])) / shoulder_width
        elbow_diff = abs(float(lm["LEFT_ELBOW"]["y"]) - float(lm["RIGHT_ELBOW"]["y"])) / shoulder_width
        # Passive monitoring cannot confirm clinical arm drift. This is only an asymmetry cue.
        normalized = 0.7 * wrist_diff + 0.3 * elbow_diff
        return min(1.0, max(0.0, (normalized - 0.22) / 0.45))

    def _center(self, lm: dict[str, dict[str, Any]]) -> tuple[float, float] | None:
        names = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in names):
            return None
        shoulders = midpoint(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"])
        hips = midpoint(lm["LEFT_HIP"], lm["RIGHT_HIP"])
        center = midpoint(shoulders, hips)
        return float(center["x"]), float(center["y"])

    def _motion_metrics(self) -> tuple[float, float, float]:
        samples: list[tuple[int, float, float]] = []
        for observation in self.history:
            center = self._center(self._landmarks(observation))
            if center:
                samples.append((int(observation.get("timestamp_ms", 0)), center[0], center[1]))
        if len(samples) < 3:
            return 0.0, 0.0, 0.0
        movements, downward_speeds = [], []
        for previous, current in zip(samples, samples[1:]):
            dt = max((current[0] - previous[0]) / 1000.0, 1e-3)
            movements.append(math.hypot(current[1] - previous[1], current[2] - previous[2]) / dt)
            downward_speeds.append((current[2] - previous[2]) / dt)
        total_drop = samples[-1][2] - samples[0][2]
        return statistics.median(movements), max(downward_speeds), total_drop

    @staticmethod
    def _persistent(states: collections.deque[SignalState], attr: str, required: int) -> bool:
        return sum(bool(getattr(state, attr)) for state in states) >= required

    def update(self, observation: dict[str, Any], pose_label: str, pose_confidence: float) -> dict[str, Any]:
        timestamp_ms = int(observation.get("timestamp_ms", 0))
        landmarks = self._landmarks(observation)
        if timestamp_ms:
            self.history.append(observation)
            self._prune(timestamp_ms)

        face_score = self._face_asymmetry_score(landmarks) if landmarks else 0.0
        arm_score = self._arm_asymmetry_score(landmarks) if landmarks else 0.0
        movement_speed, max_downward_speed, total_drop = self._motion_metrics()

        collapse_signal = (
            (pose_label in {"FALLING", "COLLAPSING"} and pose_confidence >= 0.72)
            or (max_downward_speed > 0.32 and total_drop > 0.12)
        )
        if collapse_signal:
            self.last_collapse_timestamp_ms = timestamp_ms
        freeze_after_collapse = bool(
            self.last_collapse_timestamp_ms
            and timestamp_ms - self.last_collapse_timestamp_ms >= 2500
            and timestamp_ms - self.last_collapse_timestamp_ms <= 15000
            and movement_speed < 0.012
            and pose_label in {"LYING", "UNKNOWN"}
        )

        recent_centers = []
        for obs in list(self.history)[-10:]:
            center = self._center(self._landmarks(obs))
            if center:
                recent_centers.append(center[1])
        deterioration = len(recent_centers) >= 6 and recent_centers[-1] - recent_centers[0] > 0.08 and pose_label not in {"FALLING", "COLLAPSING"}

        raw = SignalState(
            facial_droop=face_score >= 0.78,
            arm_weakness=arm_score >= 0.76,
            sudden_collapse=collapse_signal,
            pose_freeze=freeze_after_collapse,
            gradual_deterioration=deterioration,
        )
        self.boolean_history.append(raw)

        facial = self._persistent(self.boolean_history, "facial_droop", 5)
        arm = self._persistent(self.boolean_history, "arm_weakness", 5)
        collapse = self._persistent(self.boolean_history, "sudden_collapse", 2) or freeze_after_collapse
        freeze = self._persistent(self.boolean_history, "pose_freeze", 2)
        gradual = self._persistent(self.boolean_history, "gradual_deterioration", 4)

        risk_score = 0
        risk_score += 24 if facial else 0
        risk_score += 22 if arm else 0
        risk_score += 48 if collapse else 0
        risk_score += 28 if freeze else 0
        risk_score += 16 if gradual else 0
        if collapse and freeze:
            risk_score += 22
        risk_score = min(100, risk_score)

        if collapse and freeze or risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 55:
            risk_level = "HIGH"
        elif risk_score >= 25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        signs = [facial, arm, collapse, freeze, gradual]
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "facial_droop": facial,
            "arm_weakness": arm,
            "sudden_collapse": collapse,
            "pose_freeze": freeze,
            "gradual_deterioration": gradual,
            "signs_count": sum(signs),
            "evidence": {
                "face_asymmetry_score": round(face_score, 3),
                "arm_asymmetry_score": round(arm_score, 3),
                "movement_speed": round(movement_speed, 4),
                "max_downward_speed": round(max_downward_speed, 4),
                "total_drop": round(total_drop, 4),
            },
            "limitations": "Camera-based warning only; not a stroke diagnosis and does not assess speech, vision, or clinical history.",
        }
