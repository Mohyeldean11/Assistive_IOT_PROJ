from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pose_features import FEATURE_COLUMNS, FEATURE_VERSION, LEGACY_FEATURE_COLUMNS, clamp, smoothstep, vectorize, vectorize_legacy

LOGGER = logging.getLogger("careagent.pose")
ALLOWED_LABELS = {"UNKNOWN", "STANDING", "SITTING", "LYING", "FALLING", "COLLAPSING"}
LABEL_ALIASES = {
    "NONE": "UNKNOWN",
    "INIT": "UNKNOWN",
    "NOT_FOUND": "UNKNOWN",
    "LAYING": "LYING",
    "LAYING_ON_FLOOR": "LYING",
    "LAYING_ON_THE_FLOOR": "LYING",
    "LAYING_ON_BED": "LYING",
    "LAYING_ON_THE_BED": "LYING",
}


def normalize_pose_label(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    label = str(value).strip().upper().replace(" ", "_")
    label = LABEL_ALIASES.get(label, label)
    return label if label in ALLOWED_LABELS else "UNKNOWN"


class TemporalPoseStabilizer:
    def __init__(self, window: int = 5, minimum_votes: int = 2):
        self.history: collections.deque[tuple[str, float]] = collections.deque(maxlen=window)
        self.minimum_votes = minimum_votes
        self.stable_label = "UNKNOWN"
        self.stable_confidence = 0.0
        self.unknown_streak = 0

    def update(self, label: str, confidence: float, urgent: bool = False) -> tuple[str, float]:
        label = normalize_pose_label(label)
        confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        self.history.append((label, confidence))

        if label == "UNKNOWN":
            self.unknown_streak += 1
        else:
            self.unknown_streak = 0
            # Do not upload UNKNOWN/None on the first valid frame. The next frames
            # still provide smoothing and can replace this initial accepted state.
            if self.stable_label == "UNKNOWN" and confidence >= 0.60:
                self.stable_label, self.stable_confidence = label, confidence

        if self.unknown_streak >= 8:
            self.stable_label, self.stable_confidence = "UNKNOWN", 0.0
            return self.stable_label, self.stable_confidence
        if urgent and label in {"FALLING", "COLLAPSING"} and confidence >= 0.72:
            self.stable_label, self.stable_confidence = label, confidence
            return self.stable_label, self.stable_confidence

        scores: dict[str, float] = collections.defaultdict(float)
        counts: dict[str, int] = collections.defaultdict(int)
        for candidate, score in self.history:
            if candidate != "UNKNOWN":
                scores[candidate] += score
                counts[candidate] += 1
        if not scores:
            return self.stable_label, self.stable_confidence
        best = max(scores, key=lambda key: scores[key])
        if counts[best] >= self.minimum_votes:
            self.stable_label = best
            self.stable_confidence = min(1.0, scores[best] / counts[best])
        return self.stable_label, self.stable_confidence


class PoseClassifier:
    def __init__(
        self,
        model_path: Path,
        min_confidence: float = 0.66,
        floor_y_normalized: float = 0.0,
        legacy_model_path: Path | None = None,
    ):
        self.model_path = Path(model_path)
        self.legacy_model_path = Path(legacy_model_path) if legacy_model_path else None
        self.min_confidence = min_confidence
        self.floor_y_normalized = floor_y_normalized
        self.bundle: dict[str, Any] | None = None
        self.legacy_model: Any = None
        self.stabilizer = TemporalPoseStabilizer()
        self._load_models()

    def _load_models(self) -> None:
        candidates = [self.model_path]
        if self.legacy_model_path is not None and self.legacy_model_path not in candidates:
            candidates.append(self.legacy_model_path)

        for path in candidates:
            if not path.exists():
                continue
            try:
                loaded = joblib.load(path)
            except Exception:
                LOGGER.exception("Could not load pose model: %s", path)
                continue

            if isinstance(loaded, dict) and "model" in loaded:
                feature_names = loaded.get("feature_names", FEATURE_COLUMNS)
                feature_version = str(loaded.get("feature_version", FEATURE_VERSION))
                if feature_names == FEATURE_COLUMNS and feature_version == FEATURE_VERSION:
                    self.bundle = loaded
                    LOGGER.info("Loaded current pose model: %s", path)
                    return
                LOGGER.warning(
                    "Pose model at %s has an unsupported schema/version: feature_version=%s",
                    path,
                    feature_version,
                )
                continue

            if hasattr(loaded, "predict"):
                n_features = int(getattr(loaded, "n_features_in_", 0) or 0)
                if n_features == len(FEATURE_COLUMNS):
                    self.bundle = {"model": loaded, "feature_names": FEATURE_COLUMNS}
                    LOGGER.info("Loaded unbundled current-schema pose model: %s", path)
                    return
                if n_features == len(LEGACY_FEATURE_COLUMNS):
                    self.legacy_model = loaded
                    LOGGER.info("Loaded original 17-feature pose_model.pkl: %s", path)
                    return
                LOGGER.warning(
                    "Ignoring incompatible pose model %s: expected %s or %s features, got %s",
                    path,
                    len(FEATURE_COLUMNS),
                    len(LEGACY_FEATURE_COLUMNS),
                    n_features,
                )

        LOGGER.warning("No compatible trained pose classifier found; deterministic pose rules remain active")

    def _predict_with_model(self, model: Any, values: np.ndarray) -> tuple[str, float]:
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(values)[0]
                index = int(np.argmax(probabilities))
                classes = getattr(model, "classes_", [])
                label = normalize_pose_label(classes[index] if len(classes) > index else model.predict(values)[0])
                confidence = float(probabilities[index])
                return label, confidence
            label = normalize_pose_label(model.predict(values)[0])
            return label, 1.0 if label != "UNKNOWN" else 0.0
        except Exception:
            LOGGER.exception("Pose model prediction failed; using deterministic rules")
            return "UNKNOWN", 0.0

    def _model_prediction(
        self,
        features: dict[str, float],
        legacy_features: dict[str, float] | None = None,
    ) -> tuple[str, float]:
        if self.bundle and features:
            feature_names = self.bundle.get("feature_names", FEATURE_COLUMNS)
            if feature_names != FEATURE_COLUMNS:
                LOGGER.warning("Pose model feature schema mismatch; model prediction disabled")
                return "UNKNOWN", 0.0
            values = np.asarray([vectorize(features)], dtype=float)
            return self._predict_with_model(self.bundle["model"], values)

        if self.legacy_model is not None and legacy_features:
            values = np.asarray([vectorize_legacy(legacy_features)], dtype=float)
            return self._predict_with_model(self.legacy_model, values)

        return "UNKNOWN", 0.0

    def _floor_contact_score(self, f: dict[str, float]) -> float:
        if self.floor_y_normalized <= 0:
            return 0.0
        ankle_y = float(f.get("ankle_y", 0.0) or 0.0)
        # y increases downward in MediaPipe normalized image coordinates. When a
        # calibrated floor line is available, ankles close to that line strengthen
        # lying/collapse decisions but never create an emergency by themselves.
        return smoothstep(self.floor_y_normalized - 0.18, self.floor_y_normalized - 0.03, ankle_y)

    def _rules(self, f: dict[str, float]) -> tuple[str, float, bool]:
        if not f or f.get("pose_quality", 0.0) < 0.18 or f.get("visible_fraction", 0.0) < 0.40:
            return "UNKNOWN", 0.0, False

        horizontal_strength = max(
            float(f.get("horizontal_score", 0.0) or 0.0),
            smoothstep(0.95, 1.75, float(f.get("bbox_aspect_ratio", 0.0) or 0.0)),
            smoothstep(45.0, 78.0, float(f.get("torso_tilt_deg", 0.0) or 0.0)),
        )
        rapid_drop_score = max(
            smoothstep(0.25, 0.80, float(f.get("recent_drop_norm", 0.0) or 0.0)),
            smoothstep(0.55, 1.75, float(f.get("vertical_velocity_norm_s", 0.0) or 0.0)),
        )
        movement_score = smoothstep(0.035, 0.14, float(f.get("movement_norm", 0.0) or 0.0))
        floor_score = self._floor_contact_score(f)

        if rapid_drop_score >= 0.48 and movement_score >= 0.35:
            confidence = clamp(0.68 + 0.16 * rapid_drop_score + 0.10 * movement_score, 0.0, 0.97)
            return "FALLING", confidence, True
        if horizontal_strength >= 0.60 and rapid_drop_score >= 0.32:
            confidence = clamp(0.70 + 0.15 * horizontal_strength + 0.10 * rapid_drop_score + 0.05 * floor_score, 0.0, 0.96)
            return "COLLAPSING", confidence, True
        if horizontal_strength >= 0.60:
            confidence = clamp(0.61 + 0.23 * horizontal_strength + 0.04 * floor_score, 0.0, 0.94)
            return "LYING", confidence, False

        left_knee = float(f.get("left_knee_angle", 0.0) or 0.0)
        right_knee = float(f.get("right_knee_angle", 0.0) or 0.0)
        left_hip = float(f.get("left_hip_angle", 0.0) or 0.0)
        right_hip = float(f.get("right_hip_angle", 0.0) or 0.0)
        known_knees = [angle for angle in (left_knee, right_knee) if angle > 1.0]
        known_hips = [angle for angle in (left_hip, right_hip) if angle > 1.0]
        mean_knee = sum(known_knees) / len(known_knees) if known_knees else 180.0
        mean_hip = sum(known_hips) / len(known_hips) if known_hips else 180.0
        compact_upright = float(f.get("vertical_span_norm", 0.0) or 0.0) < 2.15
        flexed_legs = mean_knee < 138 or mean_hip < 130
        if flexed_legs or compact_upright:
            confidence = 0.67 + (0.06 if flexed_legs else 0.0) + (0.04 if compact_upright else 0.0)
            return "SITTING", min(0.84, confidence), False

        upright_score = min(
            smoothstep(1.85, 2.85, float(f.get("vertical_span_norm", 0.0) or 0.0)),
            1.0 - smoothstep(28.0, 55.0, float(f.get("torso_tilt_deg", 0.0) or 0.0)),
            1.0 - smoothstep(0.78, 1.22, float(f.get("bbox_aspect_ratio", 0.0) or 0.0)),
        )
        if upright_score >= 0.42:
            return "STANDING", clamp(0.66 + 0.14 * upright_score, 0.0, 0.86), False
        # Preserve the original project's practical fallback: an upright,
        # non-horizontal detected person is most likely standing rather than None.
        if float(f.get("torso_tilt_deg", 0.0) or 0.0) < 48 and float(f.get("bbox_aspect_ratio", 0.0) or 0.0) < 0.95:
            return "STANDING", 0.62, False
        return "UNKNOWN", 0.0, False

    def classify(
        self,
        features: dict[str, float],
        legacy_features: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        rule_label, rule_confidence, urgent = self._rules(features)
        model_label, model_confidence = self._model_prediction(features, legacy_features)

        if model_confidence >= self.min_confidence and model_label == rule_label and model_label != "UNKNOWN":
            raw_label = model_label
            raw_confidence = min(0.99, (model_confidence + rule_confidence) / 2.0 + 0.08)
            source = "model+rules"
        elif model_confidence >= 0.80 and rule_label == "UNKNOWN":
            raw_label, raw_confidence, source = model_label, model_confidence, "model"
        elif rule_confidence >= 0.60:
            raw_label, raw_confidence, source = rule_label, rule_confidence, "rules"
        elif model_confidence >= self.min_confidence:
            raw_label, raw_confidence, source = model_label, model_confidence, "model"
        else:
            raw_label, raw_confidence, source = "UNKNOWN", max(model_confidence, rule_confidence), "rejected"

        raw_label = normalize_pose_label(raw_label)
        stable_label, stable_confidence = self.stabilizer.update(raw_label, raw_confidence, urgent=urgent)
        stable_label = normalize_pose_label(stable_label)
        return {
            "label": stable_label,
            "confidence": round(stable_confidence, 3),
            "raw_label": raw_label,
            "raw_confidence": round(raw_confidence, 3),
            "source": source,
            "model_candidate": normalize_pose_label(model_label),
            "model_confidence": round(model_confidence, 3),
            "rule_candidate": normalize_pose_label(rule_label),
            "rule_confidence": round(rule_confidence, 3),
        }
