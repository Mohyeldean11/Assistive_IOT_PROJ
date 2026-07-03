from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass
from typing import Any

from pose_features import clamp, distance, midpoint, safe_ratio, visible


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

    @staticmethod
    def _value(point: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(point.get(key, default))
        except (TypeError, ValueError, AttributeError):
            return default
        return value if math.isfinite(value) else default

    def _prune(self, newest_timestamp_ms: int) -> None:
        cutoff = newest_timestamp_ms - int(self.history_seconds * 1000)
        while self.history and int(self.history[0].get("timestamp_ms", 0)) < cutoff:
            self.history.popleft()

    def _body_scale(self, lm: dict[str, dict[str, Any]]) -> float:
        names = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in names):
            return 0.0
        shoulders = midpoint(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"])
        hips = midpoint(lm["LEFT_HIP"], lm["RIGHT_HIP"])
        torso = distance(shoulders, hips, include_z=False)
        shoulder_width = distance(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"], include_z=False)
        hip_width = distance(lm["LEFT_HIP"], lm["RIGHT_HIP"], include_z=False)
        values = [value for value in (torso, shoulder_width * 1.35, hip_width * 1.60) if value > 1e-5]
        return max(statistics.median(values), 1e-4) if values else 0.0

    def _frontal_face(self, lm: dict[str, dict[str, Any]]) -> bool:
        needed = ["LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in needed):
            return False
        shoulder_depth_diff = abs(self._value(lm["LEFT_SHOULDER"], "z") - self._value(lm["RIGHT_SHOULDER"], "z"))
        eye_width = distance(lm["LEFT_EYE"], lm["RIGHT_EYE"], include_z=False)
        ear_width = distance(lm["LEFT_EAR"], lm["RIGHT_EAR"], include_z=False)
        eye_slope = safe_ratio(abs(self._value(lm["RIGHT_EYE"], "y") - self._value(lm["LEFT_EYE"], "y")), eye_width)
        return eye_width > 0.018 and ear_width > eye_width * 1.10 and shoulder_depth_diff < 0.24 and eye_slope < 0.45

    def _face_asymmetry_score(self, lm: dict[str, dict[str, Any]]) -> float:
        names = ["MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_EYE", "RIGHT_EYE"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in names) or not self._frontal_face(lm):
            return 0.0
        face_scale = max(distance(lm["LEFT_EYE"], lm["RIGHT_EYE"], include_z=False), 0.015)
        mouth_tilt = self._value(lm["MOUTH_RIGHT"], "y") - self._value(lm["MOUTH_LEFT"], "y")
        eye_tilt = self._value(lm["RIGHT_EYE"], "y") - self._value(lm["LEFT_EYE"], "y")
        # Compensate for head tilt: under normal tilt, the mouth and eye line move
        # in the same direction. A residual mouth-only tilt is more suspicious.
        tilt_residual = abs(mouth_tilt - eye_tilt)
        normalized = tilt_residual / face_scale
        return clamp((normalized - 0.08) / 0.28)

    def _arm_asymmetry_score(self, lm: dict[str, dict[str, Any]]) -> float:
        names = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in names):
            return 0.0
        scale = max(self._body_scale(lm), distance(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"], include_z=False), 0.03)
        left_wrist_drop = self._value(lm["LEFT_WRIST"], "y") - self._value(lm["LEFT_SHOULDER"], "y")
        right_wrist_drop = self._value(lm["RIGHT_WRIST"], "y") - self._value(lm["RIGHT_SHOULDER"], "y")
        left_elbow_drop = self._value(lm["LEFT_ELBOW"], "y") - self._value(lm["LEFT_SHOULDER"], "y")
        right_elbow_drop = self._value(lm["RIGHT_ELBOW"], "y") - self._value(lm["RIGHT_SHOULDER"], "y")
        wrist_residual = abs(left_wrist_drop - right_wrist_drop) / scale
        elbow_residual = abs(left_elbow_drop - right_elbow_drop) / scale
        normalized = 0.68 * wrist_residual + 0.32 * elbow_residual
        shoulder_depth_diff = abs(self._value(lm["LEFT_SHOULDER"], "z") - self._value(lm["RIGHT_SHOULDER"], "z"))
        perspective_weight = clamp(1.0 - shoulder_depth_diff / 0.35, 0.35, 1.0)
        return clamp((normalized - 0.24) / 0.52) * perspective_weight

    def _center(self, lm: dict[str, dict[str, Any]]) -> tuple[float, float, float] | None:
        names = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        if any(not visible(lm.get(name), self.visibility_threshold) for name in names):
            return None
        shoulders = midpoint(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"])
        hips = midpoint(lm["LEFT_HIP"], lm["RIGHT_HIP"])
        center = midpoint(shoulders, hips)
        scale = self._body_scale(lm)
        if scale <= 0:
            return None
        return float(center["x"]), float(center["y"]), scale

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = clamp(percentile / 100.0) * (len(ordered) - 1)
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return ordered[low]
        fraction = rank - low
        return ordered[low] * (1.0 - fraction) + ordered[high] * fraction

    def _motion_metrics(self) -> tuple[float, float, float]:
        samples: list[tuple[int, float, float, float]] = []
        for observation in self.history:
            center = self._center(self._landmarks(observation))
            if center:
                samples.append((int(observation.get("timestamp_ms", 0)), center[0], center[1], center[2]))
        if len(samples) < 3:
            return 0.0, 0.0, 0.0
        movements, downward_speeds = [], []
        for previous, current in zip(samples, samples[1:]):
            dt = max((current[0] - previous[0]) / 1000.0, 1e-3)
            scale = max(statistics.median([previous[3], current[3]]), 1e-4)
            movements.append(math.hypot(current[1] - previous[1], current[2] - previous[2]) / scale / dt)
            downward_speeds.append((current[2] - previous[2]) / scale / dt)
        median_scale = max(statistics.median([sample[3] for sample in samples]), 1e-4)
        total_drop = (samples[-1][2] - samples[0][2]) / median_scale
        positive_downward = [speed for speed in downward_speeds if speed > 0]
        return statistics.median(movements), self._percentile(positive_downward, 90), total_drop

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
            or (max_downward_speed > 1.25 and total_drop > 0.55)
        )
        if collapse_signal:
            self.last_collapse_timestamp_ms = timestamp_ms
        freeze_after_collapse = bool(
            self.last_collapse_timestamp_ms
            and timestamp_ms - self.last_collapse_timestamp_ms >= 2500
            and timestamp_ms - self.last_collapse_timestamp_ms <= 15000
            and movement_speed < 0.05
            and pose_label in {"LYING", "UNKNOWN"}
        )

        recent_centers: list[tuple[float, float]] = []
        for obs in list(self.history)[-10:]:
            center = self._center(self._landmarks(obs))
            if center:
                recent_centers.append((center[1], center[2]))
        deterioration = False
        if len(recent_centers) >= 6:
            median_scale = max(statistics.median([center[1] for center in recent_centers]), 1e-4)
            deterioration = (recent_centers[-1][0] - recent_centers[0][0]) / median_scale > 0.35 and pose_label not in {"FALLING", "COLLAPSING"}

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
                "movement_speed_body_lengths_s": round(movement_speed, 4),
                "max_downward_speed_body_lengths_s": round(max_downward_speed, 4),
                "total_drop_body_lengths": round(total_drop, 4),
            },
            "limitations": "Camera-based warning only; not a stroke diagnosis and does not assess speech, vision, or clinical history.",
        }
