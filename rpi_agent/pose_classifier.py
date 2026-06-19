from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pose_features import FEATURE_COLUMNS, LEGACY_FEATURE_COLUMNS, vectorize, vectorize_legacy

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
                if feature_names == FEATURE_COLUMNS:
                    self.bundle = loaded
                    LOGGER.info("Loaded current pose model: %s", path)
                    return
                LOGGER.warning("Pose model at %s has an unsupported feature schema", path)
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

    def _rules(self, f: dict[str, float]) -> tuple[str, float, bool]:
        if not f or f.get("pose_quality", 0.0) < 0.18 or f.get("visible_fraction", 0.0) < 0.40:
            return "UNKNOWN", 0.0, False
        horizontal = f["bbox_aspect_ratio"] > 1.05 or f["horizontal_score"] > 0.62 or f["torso_tilt_deg"] > 58
        rapid_drop = f["recent_drop_norm"] > 0.38 or f["vertical_velocity_norm_s"] > 0.75
        moving = f["movement_norm"] > 0.07
        if rapid_drop and moving:
            return "FALLING", min(0.96, 0.68 + 0.18 * min(1.0, f["recent_drop_norm"])), True
        if horizontal and rapid_drop:
            return "COLLAPSING", 0.78, True
        if horizontal:
            confidence = min(0.92, 0.62 + 0.2 * min(1.0, f["horizontal_score"]))
            return "LYING", confidence, False
        mean_knee = (f["left_knee_angle"] + f["right_knee_angle"]) / 2.0
        mean_hip = (f["left_hip_angle"] + f["right_hip_angle"]) / 2.0
        if mean_knee < 135 or mean_hip < 125 or f["vertical_span_norm"] < 2.25:
            return "SITTING", 0.70, False
        if f["vertical_span_norm"] >= 2.0 and f["torso_tilt_deg"] < 40:
            return "STANDING", 0.75, False
        # Preserve the original project's practical fallback: an upright,
        # non-horizontal detected person is most likely standing rather than None.
        if f["torso_tilt_deg"] < 48 and f["bbox_aspect_ratio"] < 0.95:
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
