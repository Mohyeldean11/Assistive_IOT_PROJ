from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

FEATURE_VERSION = "2.1"
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
TORSO_LANDMARKS = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _value(point: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = point.get(key, default)
    except AttributeError:
        return default
    return _finite(value, default)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _finite(value, low)))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    denominator = _finite(denominator)
    if abs(denominator) < 1e-6:
        return default
    return _finite(numerator) / denominator


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    """Return a smooth 0..1 score between two thresholds."""
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if value >= edge1 else 0.0
    x = clamp((value - edge0) / (edge1 - edge0))
    return x * x * (3.0 - 2.0 * x)


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
    cosine = clamp((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _angle_between_3d(a: dict[str, float], b: dict[str, float]) -> float:
    na = math.sqrt(sum(a[k] ** 2 for k in ("x", "y", "z")))
    nb = math.sqrt(sum(b[k] ** 2 for k in ("x", "y", "z")))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    dot = sum(a[k] * b[k] for k in ("x", "y", "z"))
    return math.degrees(math.acos(clamp(dot / (na * nb), -1.0, 1.0)))


def _vector(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {axis: _value(b, axis) - _value(a, axis) for axis in ("x", "y", "z")}


def robust_median(values: Iterable[float], default: float = 0.0) -> float:
    cleaned = [_finite(v) for v in values if math.isfinite(_finite(v))]
    return statistics.median(cleaned) if cleaned else default


def robust_percentile(values: Iterable[float], percentile: float, default: float = 0.0) -> float:
    cleaned = sorted(_finite(v) for v in values if math.isfinite(_finite(v)))
    if not cleaned:
        return default
    if len(cleaned) == 1:
        return cleaned[0]
    rank = clamp(percentile / 100.0) * (len(cleaned) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return cleaned[low]
    fraction = rank - low
    return cleaned[low] * (1.0 - fraction) + cleaned[high] * fraction


def _frame_landmarks(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    landmarks = observation.get("landmarks", observation)
    return landmarks if isinstance(landmarks, dict) else {}


def _visible_points(
    landmarks: dict[str, dict[str, Any]],
    names: Sequence[str],
    threshold: float,
) -> list[dict[str, Any]]:
    return [landmarks[name] for name in names if visible(landmarks.get(name), threshold)]


def _visibility_strength(
    landmarks: dict[str, dict[str, Any]],
    names: Sequence[str],
    threshold: float,
) -> float:
    scores = []
    for name in names:
        point = landmarks.get(name)
        if not point:
            scores.append(0.0)
            continue
        visibility = clamp(_value(point, "visibility", 0.0))
        presence_raw = point.get("presence")
        presence = visibility if presence_raw is None else clamp(_value(point, "presence", visibility))
        if presence <= 0.0 and visibility > 0.0:
            presence = visibility
        # Keep a little tolerance below the hard threshold so borderline but stable
        # landmarks reduce confidence instead of disappearing abruptly.
        scores.append(clamp((0.75 * visibility + 0.25 * presence) / max(threshold, 0.15)))
    return sum(scores) / len(scores) if scores else 0.0


def _body_scale(landmarks: dict[str, dict[str, Any]], threshold: float = 0.6) -> float:
    """Robust body scale in normalized image units.

    Uses torso length plus shoulder/hip width so the scale remains usable when a
    person is partly turned or transitioning between upright and horizontal poses.
    """
    if any(name not in landmarks for name in TORSO_LANDMARKS):
        return 0.0
    ls, rs = landmarks["LEFT_SHOULDER"], landmarks["RIGHT_SHOULDER"]
    lh, rh = landmarks["LEFT_HIP"], landmarks["RIGHT_HIP"]
    if sum(visible(landmarks.get(name), threshold) for name in TORSO_LANDMARKS) < 3:
        return 0.0
    mid_shoulders = midpoint(ls, rs)
    mid_hips = midpoint(lh, rh)
    candidates = [
        distance(mid_shoulders, mid_hips, include_z=False),
        distance(ls, rs, include_z=False) * 1.35,
        distance(lh, rh, include_z=False) * 1.60,
    ]
    return max(robust_median([c for c in candidates if c > 1e-5], default=0.0), 1e-4)


def _angle_if_visible(
    landmarks: dict[str, dict[str, Any]],
    a: str,
    vertex: str,
    c: str,
    threshold: float,
) -> float:
    if all(visible(landmarks.get(name), threshold) for name in (a, vertex, c)):
        return angle(landmarks[a], landmarks[vertex], landmarks[c])
    return 0.0


def _mean_y(points: Sequence[dict[str, Any]], default: float = 0.0) -> float:
    return robust_median((_value(point, "y") for point in points), default)


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
    latest_scale = _body_scale(latest, visibility_threshold) or torso_length

    usable_points = [point for point in latest.values() if visible(point, visibility_threshold)]
    xs = [_value(point, "x") for point in usable_points]
    ys = [_value(point, "y") for point in usable_points]
    bbox_width = max(xs) - min(xs) if xs else 0.0
    bbox_height = max(ys) - min(ys) if ys else 0.0
    bbox_aspect_ratio = safe_ratio(bbox_width, max(bbox_height, 1e-4))

    upper_points = _visible_points(latest, ("NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_SHOULDER", "RIGHT_SHOULDER"), visibility_threshold)
    lower_points = _visible_points(latest, ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP"), visibility_threshold)
    if upper_points and lower_points:
        vertical_span = max(_value(point, "y") for point in lower_points) - min(_value(point, "y") for point in upper_points)
    else:
        vertical_span = max(_value(la, "y"), _value(ra, "y")) - min(_value(nose, "y"), _value(ls, "y"), _value(rs, "y"))
    vertical_span_norm = max(0.0, vertical_span) / latest_scale

    dx = _value(mh, "x") - _value(ms, "x")
    dy = _value(mh, "y") - _value(ms, "y")
    torso_tilt_deg = abs(math.degrees(math.atan2(dx, max(abs(dy), 1e-5))))
    aspect_score = smoothstep(0.85, 1.70, bbox_aspect_ratio)
    tilt_score = smoothstep(35.0, 75.0, torso_tilt_deg)
    compact_span_score = 1.0 - smoothstep(1.35, 2.55, vertical_span_norm)
    horizontal_score = clamp(0.50 * aspect_score + 0.35 * tilt_score + 0.15 * compact_span_score)

    left_knee_angle = _angle_if_visible(latest, "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", visibility_threshold)
    right_knee_angle = _angle_if_visible(latest, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", visibility_threshold)
    left_hip_angle = _angle_if_visible(latest, "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE", visibility_threshold)
    right_hip_angle = _angle_if_visible(latest, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", visibility_threshold)
    head_hip_dx_norm = safe_ratio(abs(_value(nose, "x") - _value(mh, "x")), latest_scale) if visible(nose, visibility_threshold) else 0.0

    lw = latest.get("LEFT_WRIST", {})
    rw = latest.get("RIGHT_WRIST", {})
    wrist_height_diff_norm = safe_ratio(abs(_value(lw, "y") - _value(rw, "y")), latest_scale) if visible(lw, visibility_threshold) and visible(rw, visibility_threshold) else 0.0

    centers: list[tuple[float, float, float, float]] = []
    for obs in valid_observations:
        lm = _frame_landmarks(obs)
        if all(name in lm for name in TORSO_LANDMARKS):
            scale = _body_scale(lm, visibility_threshold)
            if scale <= 0:
                continue
            c1 = midpoint(lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"])
            c2 = midpoint(lm["LEFT_HIP"], lm["RIGHT_HIP"])
            center = midpoint(c1, c2)
            centers.append((_value(center, "x"), _value(center, "y"), float(obs.get("timestamp_ms", len(centers) * 100)), scale))

    movement_steps = []
    velocity_steps = []
    for previous, current in zip(centers, centers[1:]):
        step_scale = max(robust_median([previous[3], current[3]], default=latest_scale), 1e-4)
        movement_steps.append(math.hypot(current[0] - previous[0], current[1] - previous[1]) / step_scale)
        dt = max((current[2] - previous[2]) / 1000.0, 1e-3)
        velocity_steps.append((current[1] - previous[1]) / step_scale / dt)
    movement_norm = robust_percentile(movement_steps, 75, default=0.0)
    recent_drop_norm = safe_ratio(centers[-1][1] - centers[0][1], robust_median([c[3] for c in centers], default=latest_scale)) if len(centers) >= 2 else 0.0
    positive_velocity = [value for value in velocity_steps if value > 0]
    vertical_velocity_norm_s = robust_percentile(positive_velocity, 90, default=0.0)

    depths = [_value(point, "z") for point in usable_points]
    depth_spread_norm = safe_ratio(max(depths) - min(depths), latest_scale) if depths else 0.0
    visibility_strength = _visibility_strength(latest, CORE_LANDMARKS, visibility_threshold)
    shape_score = clamp(1.0 - abs(bbox_width - 0.42) / 0.62)
    scale_score = smoothstep(0.035, 0.110, torso_length)
    pose_quality = clamp(0.45 * visibility_strength + 0.35 * visible_fraction + 0.20 * shape_score) * max(0.55, scale_score)

    hip_points = _visible_points(latest, ("LEFT_HIP", "RIGHT_HIP"), visibility_threshold) or [lh, rh]
    ankle_points = _visible_points(latest, ("LEFT_ANKLE", "RIGHT_ANKLE"), visibility_threshold)
    if not ankle_points:
        ankle_points = _visible_points(latest, ("LEFT_KNEE", "RIGHT_KNEE"), visibility_threshold) or [la, ra]

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
        "hip_y": _mean_y(hip_points, (_value(lh, "y") + _value(rh, "y")) / 2),
        "ankle_y": _mean_y(ankle_points, (_value(la, "y") + _value(ra, "y")) / 2),
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
