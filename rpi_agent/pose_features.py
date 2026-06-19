from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

FEATURE_VERSION = "2.0"
FEATURE_COLUMNS = [
    "pose_quality", "visible_fraction", "torso_length", "shoulder_width", "hip_width",
    "bbox_aspect_ratio", "vertical_span_norm", "torso_tilt_deg", "horizontal_score",
    "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
    "head_hip_dx_norm", "wrist_height_diff_norm", "movement_norm", "recent_drop_norm",
    "vertical_velocity_norm_s", "depth_spread_norm", "hip_y", "ankle_y", "body_center_y",
]
LEGACY_FEATURE_COLUMNS = [
    "shoulder_distance", "hip_distance", "knee_distance", "vertical_span", "spine_drop",
    "is_horizontal", "depth_spread", "movement_score", "recent_drop", "shoulder_hip_height",
    "torso_angle", "left_leg_angle", "right_leg_angle", "head_hip_ratio",
    "shoulder_knee_ratio", "hip_knee_ratio", "head_slope",
]
CORE_LANDMARKS = (
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
)


def _value(point: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = point.get(key, default)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def visible(point: dict[str, Any] | None, threshold: float = 0.6) -> bool:
    """Return whether a landmark is usable across MediaPipe versions.

    ``presence`` may be unset by a model/build. Visibility is therefore the primary
    signal. Presence is only used as an additional check when it is meaningfully set.
    """
    if not point:
        return False
    visibility = _value(point, "visibility", 1.0)
    presence_raw = point.get("presence")
    if presence_raw is None:
        return visibility >= threshold
    presence = _value(point, "presence", visibility)
    if presence <= 0.0 and visibility > 0.0:
        return visibility >= threshold
    return visibility >= threshold and presence >= min(0.35, threshold)


def distance(a: dict[str, Any], b: dict[str, Any], include_z: bool = True) -> float:
    dz = (_value(a, "z") - _value(b, "z")) if include_z else 0.0
    return math.sqrt((_value(a, "x") - _value(b, "x")) ** 2 + (_value(a, "y") - _value(b, "y")) ** 2 + dz**2)


def midpoint(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {axis: (_value(a, axis) + _value(b, axis)) / 2.0 for axis in ("x", "y", "z")}


def angle(a: dict[str, Any], vertex: dict[str, Any], c: dict[str, Any]) -> float:
    v1 = (_value(a, "x") - _value(vertex, "x"), _value(a, "y") - _value(vertex, "y"))
    v2 = (_value(c, "x") - _value(vertex, "x"), _value(c, "y") - _value(vertex, "y"))
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosine))


def _angle_between_3d(a: dict[str, float], b: dict[str, float]) -> float:
    na = math.sqrt(sum(a[k] ** 2 for k in ("x", "y", "z")))
    nb = math.sqrt(sum(b[k] ** 2 for k in ("x", "y", "z")))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    dot = sum(a[k] * b[k] for k in ("x", "y", "z"))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (na * nb)))))


def _vector(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {axis: _value(b, axis) - _value(a, axis) for axis in ("x", "y", "z")}


def robust_median(values: Iterable[float], default: float = 0.0) -> float:
    cleaned = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(cleaned) if cleaned else default


def _frame_landmarks(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    landmarks = observation.get("landmarks", observation)
    return landmarks if isinstance(landmarks, dict) else {}


def compute_features(observations: list[dict[str, Any]], visibility_threshold: float = 0.6) -> dict[str, float]:
    valid_observations = [obs for obs in observations if _frame_landmarks(obs)]
    if not valid_observations:
        return {}
    latest = _frame_landmarks(valid_observations[-1])
    if any(name not in latest for name in CORE_LANDMARKS):
        return {}

    visible_count = sum(visible(latest.get(name), visibility_threshold) for name in CORE_LANDMARKS)
    visible_fraction = visible_count / len(CORE_LANDMARKS)
    # Four visible core landmarks are enough to avoid discarding a cropped but valid body.
    if visible_fraction < 4 / len(CORE_LANDMARKS):
        return {}

    ls, rs = latest["LEFT_SHOULDER"], latest["RIGHT_SHOULDER"]
    lh, rh = latest["LEFT_HIP"], latest["RIGHT_HIP"]
    lk, rk = latest["LEFT_KNEE"], latest["RIGHT_KNEE"]
    la, ra = latest["LEFT_ANKLE"], latest["RIGHT_ANKLE"]
    nose = latest["NOSE"]
    ms, mh = midpoint(ls, rs), midpoint(lh, rh)

    torso_length = max(distance(ms, mh, include_z=False), 1e-4)
    shoulder_width = distance(ls, rs, include_z=False)
    hip_width = distance(lh, rh, include_z=False)

    usable_points = [p for p in latest.values() if visible(p, visibility_threshold)]
    xs = [_value(p, "x") for p in usable_points]
    ys = [_value(p, "y") for p in usable_points]
    bbox_width = max(xs) - min(xs) if xs else 0.0
    bbox_height = max(ys) - min(ys) if ys else 0.0
    bbox_aspect_ratio = bbox_width / max(bbox_height, 1e-4)

    vertical_span = max(_value(la, "y"), _value(ra, "y")) - min(_value(nose, "y"), _value(ls, "y"), _value(rs, "y"))
    vertical_span_norm = vertical_span / torso_length
    dx = _value(mh, "x") - _value(ms, "x")
    dy = _value(mh, "y") - _value(ms, "y")
    torso_tilt_deg = abs(math.degrees(math.atan2(dx, max(abs(dy), 1e-5))))
    horizontal_score = min(1.0, bbox_aspect_ratio / 1.8)

    left_knee_angle = angle(lh, lk, la)
    right_knee_angle = angle(rh, rk, ra)
    left_hip_angle = angle(ls, lh, lk)
    right_hip_angle = angle(rs, rh, rk)
    head_hip_dx_norm = abs(_value(nose, "x") - _value(mh, "x")) / torso_length

    lw = latest.get("LEFT_WRIST", {})
    rw = latest.get("RIGHT_WRIST", {})
    wrist_height_diff_norm = abs(_value(lw, "y") - _value(rw, "y")) / torso_length if visible(lw, visibility_threshold) and visible(rw, visibility_threshold) else 0.0

    centers: list[tuple[float, float, float]] = []
    for obs in valid_observations:
        lm = _frame_landmarks(obs)
        if all(name in lm for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")):
            c1 = midpoint(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"])
            c2 = midpoint(lm["LEFT_HIP"], lm["RIGHT_HIP"])
            center = midpoint(c1, c2)
            centers.append((_value(center, "x"), _value(center, "y"), float(obs.get("timestamp_ms", len(centers) * 100))))

    movement_steps = []
    velocity_steps = []
    for previous, current in zip(centers, centers[1:]):
        movement_steps.append(math.hypot(current[0] - previous[0], current[1] - previous[1]) / torso_length)
        dt = max((current[2] - previous[2]) / 1000.0, 1e-3)
        velocity_steps.append((current[1] - previous[1]) / torso_length / dt)
    movement_norm = robust_median(movement_steps)
    recent_drop_norm = (centers[-1][1] - centers[0][1]) / torso_length if len(centers) >= 2 else 0.0
    vertical_velocity_norm_s = max(velocity_steps, default=0.0)

    depths = [_value(p, "z") for p in usable_points]
    depth_spread_norm = (max(depths) - min(depths)) / torso_length if depths else 0.0
    quality_shape = max(0.35, min(1.0, 1.0 - abs(bbox_width - 0.45)))
    pose_quality = visible_fraction * quality_shape

    values = {
        "pose_quality": pose_quality,
        "visible_fraction": visible_fraction,
        "torso_length": torso_length,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "vertical_span_norm": vertical_span_norm,
        "torso_tilt_deg": torso_tilt_deg,
        "horizontal_score": horizontal_score,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "head_hip_dx_norm": head_hip_dx_norm,
        "wrist_height_diff_norm": wrist_height_diff_norm,
        "movement_norm": movement_norm,
        "recent_drop_norm": recent_drop_norm,
        "vertical_velocity_norm_s": vertical_velocity_norm_s,
        "depth_spread_norm": depth_spread_norm,
        "hip_y": (_value(lh, "y") + _value(rh, "y")) / 2,
        "ankle_y": (_value(la, "y") + _value(ra, "y")) / 2,
        "body_center_y": _value(midpoint(ms, mh), "y"),
    }
    return {name: float(values.get(name, 0.0)) for name in FEATURE_COLUMNS}


def compute_legacy_features(observations: list[dict[str, Any]]) -> dict[str, float]:
    """Reproduce the original 17-feature schema for an existing pose_model.pkl."""
    valid = [obs for obs in observations if _frame_landmarks(obs)]
    if not valid:
        return {}
    latest = _frame_landmarks(valid[-1])
    required = ("NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE")
    if any(name not in latest for name in required):
        return {}

    ls, rs = latest["LEFT_SHOULDER"], latest["RIGHT_SHOULDER"]
    lh, rh = latest["LEFT_HIP"], latest["RIGHT_HIP"]
    lk, rk = latest["LEFT_KNEE"], latest["RIGHT_KNEE"]
    la, ra = latest["LEFT_ANKLE"], latest["RIGHT_ANKLE"]
    nose = latest["NOSE"]
    avg_shoulder_y = (_value(ls, "y") + _value(rs, "y")) / 2
    avg_hip_y = (_value(lh, "y") + _value(rh, "y")) / 2
    avg_knee_y = (_value(lk, "y") + _value(rk, "y")) / 2
    shoulder_dist = distance(ls, rs)
    hip_dist = distance(lh, rh)
    knee_dist = distance(lk, rk)
    vertical_span = abs(avg_knee_y - avg_shoulder_y)
    spine_drop = abs(avg_hip_y - avg_shoulder_y)
    is_horizontal = float(abs(avg_shoulder_y - avg_hip_y) < 0.08 and abs(avg_hip_y - avg_knee_y) < 0.08)
    depths = [_value(p, "z") for p in latest.values()]
    depth_spread = max(depths, default=0.0) - min(depths, default=0.0)

    movement_scores = []
    for previous_obs, current_obs in zip(valid, valid[1:]):
        previous, current = _frame_landmarks(previous_obs), _frame_landmarks(current_obs)
        frame_movements = [distance(previous[name], current[name]) for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP") if name in previous and name in current]
        if frame_movements:
            movement_scores.append(sum(frame_movements) / len(frame_movements))
    movement_score = sum(movement_scores) / len(movement_scores) if movement_scores else 0.0
    first = _frame_landmarks(valid[0])
    start_shoulder_y = (_value(first.get("LEFT_SHOULDER", {}), "y") + _value(first.get("RIGHT_SHOULDER", {}), "y")) / 2
    recent_drop = avg_shoulder_y - start_shoulder_y

    mid_shoulder = midpoint(ls, rs)
    mid_hip = midpoint(lh, rh)
    torso_angle = _angle_between_3d(_vector(mid_shoulder, mid_hip), {"x": 0.0, "y": 1.0, "z": 0.0})
    left_leg_angle = _angle_between_3d(_vector(lh, lk), _vector(lk, la))
    right_leg_angle = _angle_between_3d(_vector(rh, rk), _vector(rk, ra))
    head_to_hip = distance(nose, mid_hip)
    head_hip_ratio = head_to_hip / vertical_span if vertical_span else 0.0
    shoulder_knee_ratio = shoulder_dist / knee_dist if knee_dist else 0.0
    hip_knee_ratio = hip_dist / knee_dist if knee_dist else 0.0
    head_slope = abs(_value(nose, "x") - _value(mid_hip, "x")) / vertical_span if vertical_span else 0.0

    values = {
        "shoulder_distance": shoulder_dist,
        "hip_distance": hip_dist,
        "knee_distance": knee_dist,
        "vertical_span": vertical_span,
        "spine_drop": spine_drop,
        "is_horizontal": is_horizontal,
        "depth_spread": depth_spread,
        "movement_score": movement_score,
        "recent_drop": recent_drop,
        "shoulder_hip_height": abs(avg_shoulder_y - avg_hip_y),
        "torso_angle": torso_angle,
        "left_leg_angle": left_leg_angle,
        "right_leg_angle": right_leg_angle,
        "head_hip_ratio": head_hip_ratio,
        "shoulder_knee_ratio": shoulder_knee_ratio,
        "hip_knee_ratio": hip_knee_ratio,
        "head_slope": head_slope,
    }
    return {name: float(values[name]) for name in LEGACY_FEATURE_COLUMNS}


def vectorize(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_COLUMNS]


def vectorize_legacy(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in LEGACY_FEATURE_COLUMNS]
