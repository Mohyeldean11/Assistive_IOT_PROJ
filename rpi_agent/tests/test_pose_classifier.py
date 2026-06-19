from pathlib import Path

from pose_classifier import PoseClassifier, normalize_pose_label
from pose_features import compute_features, visible


def base_features():
    return {
        "pose_quality": 0.9, "visible_fraction": 1.0, "bbox_aspect_ratio": 0.45,
        "horizontal_score": 0.25, "torso_tilt_deg": 5, "recent_drop_norm": 0.0,
        "vertical_velocity_norm_s": 0.0, "movement_norm": 0.01,
        "left_knee_angle": 172, "right_knee_angle": 170,
        "left_hip_angle": 171, "right_hip_angle": 170, "vertical_span_norm": 3.2,
    }


def test_first_valid_pose_is_not_none_or_unknown(tmp_path: Path):
    classifier = PoseClassifier(tmp_path / "none.joblib")
    result = classifier.classify(base_features())
    assert result["label"] == "STANDING"
    assert result["label"] is not None


def test_low_quality_rejected(tmp_path: Path):
    classifier = PoseClassifier(tmp_path / "none.joblib")
    features = base_features()
    features["pose_quality"] = 0.1
    result = classifier.classify(features)
    assert result["raw_label"] == "UNKNOWN"


def test_unknown_eventually_clears_stale_pose(tmp_path: Path):
    classifier = PoseClassifier(tmp_path / "none.joblib")
    classifier.classify(base_features())
    bad = base_features()
    bad["pose_quality"] = 0.1
    for _ in range(8):
        result = classifier.classify(bad)
    assert result["label"] == "UNKNOWN"


def test_missing_or_zero_presence_uses_visibility():
    assert visible({"visibility": 0.9}, 0.5)
    assert visible({"visibility": 0.9, "presence": 0.0}, 0.5)


def test_none_label_normalized():
    assert normalize_pose_label(None) == "UNKNOWN"
    assert normalize_pose_label("LAYING ON THE FLOOR") == "LYING"


def test_features_survive_zero_presence_values():
    coordinates = {
        "NOSE": (0.50, 0.12),
        "LEFT_SHOULDER": (0.42, 0.30), "RIGHT_SHOULDER": (0.58, 0.30),
        "LEFT_HIP": (0.45, 0.55), "RIGHT_HIP": (0.55, 0.55),
        "LEFT_KNEE": (0.46, 0.75), "RIGHT_KNEE": (0.54, 0.75),
        "LEFT_ANKLE": (0.46, 0.95), "RIGHT_ANKLE": (0.54, 0.95),
    }
    landmarks = {
        name: {"x": x, "y": y, "z": 0.0, "visibility": 0.95, "presence": 0.0}
        for name, (x, y) in coordinates.items()
    }
    result = compute_features([{"timestamp_ms": 1000, "landmarks": landmarks}], 0.5)
    assert result
    assert result["visible_fraction"] == 1.0
