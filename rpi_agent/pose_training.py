from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from config import BASE_DIR, settings
from pose_features import FEATURE_COLUMNS, FEATURE_VERSION, compute_features, vectorize

LOGGER = logging.getLogger("careagent.training")
DEFAULT_DATASET = BASE_DIR / "pose_dataset.csv"
DEFAULT_MODEL = BASE_DIR / "models" / "pose_classifier.joblib"
ALLOWED_LABELS = {"STANDING", "SITTING", "LYING", "FALLING", "COLLAPSING"}
METADATA_COLUMNS = ["sample_id", "session_id", "captured_at", "label", "feature_version", "camera"]


def normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace(" ", "_")
    aliases = {"LAYING": "LYING", "LAYING_ON_FLOOR": "LYING", "LAYING_ON_BED": "LYING"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported label {label!r}. Use one of: {', '.join(sorted(ALLOWED_LABELS))}")
    return normalized


def append_sample(path: Path, features: dict[str, float], label: str, session_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "sample_id": str(uuid.uuid4()),
        "session_id": session_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "feature_version": FEATURE_VERSION,
        "camera": "Logitech C930/C930e 90deg",
        **{name: features.get(name, 0.0) for name in FEATURE_COLUMNS},
    }
    header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS + FEATURE_COLUMNS)
        if header:
            writer.writeheader()
        writer.writerow(row)


def collect_samples(label: str, samples: int = 100, dataset_path: Path = DEFAULT_DATASET) -> None:
    from camera_pipeline import CameraPosePipeline

    label = normalize_label(label)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    session_id = f"{label.lower()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    observations: deque[dict[str, Any]] = deque(maxlen=10)
    saved = 0
    rejected = 0
    print(f"Collecting {samples} samples for {label}. Move naturally and vary orientation slightly.")
    print("Press Ctrl+C to stop. Samples with missing/low-confidence landmarks are rejected.")
    with CameraPosePipeline(settings) as camera:
        while saved < samples:
            observation = camera.read_observation()
            observations.append(observation)
            features = compute_features(list(observations), settings.min_landmark_visibility)
            if features and features.get("pose_quality", 0.0) >= settings.min_pose_quality:
                append_sample(Path(dataset_path), features, label, session_id)
                saved += 1
                print(f"\rSaved {saved}/{samples} | rejected {rejected}", end="", flush=True)
            else:
                rejected += 1
            time.sleep(max(0.12, settings.analysis_interval_seconds / 2))
    print(f"\nCollection finished: {saved} saved, {rejected} rejected.")


def _bounded_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    return float(np.clip(rng.normal(mean, sd), low, high))


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _paired_angle(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> tuple[float, float]:
    base = _bounded_normal(rng, mean, sd, low, high)
    asymmetry = _bounded_normal(rng, 0.0, sd * 0.25, -sd, sd)
    return float(np.clip(base + asymmetry, low, high)), float(np.clip(base - asymmetry, low, high))


def _synthetic_feature_row(label: str, rng: np.random.Generator, session_bias: dict[str, float]) -> dict[str, float]:
    torso_length = _bounded_normal(rng, session_bias["torso_length"], 0.025, 0.11, 0.36)
    shoulder_width = torso_length * _bounded_normal(rng, 0.72, 0.07, 0.55, 0.92)
    hip_width = torso_length * _bounded_normal(rng, 0.58, 0.06, 0.42, 0.82)
    pose_quality = _bounded_normal(rng, 0.86, 0.08, 0.48, 1.0)
    visible_fraction = _bounded_normal(rng, 0.93, 0.07, 0.66, 1.0)
    head_hip_dx_norm = _bounded_normal(rng, 0.10, 0.07, 0.0, 0.38)
    wrist_height_diff_norm = _bounded_normal(rng, 0.14, 0.11, 0.0, 0.62)
    depth_spread_norm = _bounded_normal(rng, 0.42, 0.18, 0.02, 1.15)

    if label == "STANDING":
        left_knee, right_knee = _paired_angle(rng, 171.0, 6.0, 145.0, 180.0)
        left_hip, right_hip = _paired_angle(rng, 169.0, 7.0, 140.0, 180.0)
        values = {
            "bbox_aspect_ratio": _bounded_normal(rng, 0.48, 0.10, 0.28, 0.82),
            "vertical_span_norm": _bounded_normal(rng, 3.05, 0.35, 2.15, 4.15),
            "torso_tilt_deg": _bounded_normal(rng, 9.0, 8.0, 0.0, 34.0),
            "horizontal_score": _bounded_normal(rng, 0.12, 0.08, 0.0, 0.42),
            "movement_norm": _bounded_normal(rng, 0.018, 0.018, 0.0, 0.09),
            "recent_drop_norm": _bounded_normal(rng, 0.0, 0.055, -0.14, 0.18),
            "vertical_velocity_norm_s": _bounded_normal(rng, 0.04, 0.08, 0.0, 0.36),
            "hip_y": _bounded_normal(rng, 0.55, 0.08, 0.34, 0.75),
            "ankle_y": _bounded_normal(rng, 0.91, 0.05, 0.72, 0.99),
            "body_center_y": _bounded_normal(rng, 0.43, 0.08, 0.25, 0.64),
        }
    elif label == "SITTING":
        left_knee, right_knee = _paired_angle(rng, 96.0, 18.0, 45.0, 145.0)
        left_hip, right_hip = _paired_angle(rng, 104.0, 16.0, 55.0, 148.0)
        values = {
            "bbox_aspect_ratio": _bounded_normal(rng, 0.72, 0.15, 0.38, 1.12),
            "vertical_span_norm": _bounded_normal(rng, 2.05, 0.28, 1.35, 2.65),
            "torso_tilt_deg": _bounded_normal(rng, 14.0, 12.0, 0.0, 48.0),
            "horizontal_score": _bounded_normal(rng, 0.24, 0.13, 0.0, 0.56),
            "movement_norm": _bounded_normal(rng, 0.015, 0.016, 0.0, 0.08),
            "recent_drop_norm": _bounded_normal(rng, 0.0, 0.05, -0.12, 0.16),
            "vertical_velocity_norm_s": _bounded_normal(rng, 0.03, 0.07, 0.0, 0.30),
            "hip_y": _bounded_normal(rng, 0.63, 0.08, 0.42, 0.83),
            "ankle_y": _bounded_normal(rng, 0.82, 0.09, 0.56, 0.99),
            "body_center_y": _bounded_normal(rng, 0.50, 0.08, 0.31, 0.72),
        }
    elif label == "LYING":
        left_knee, right_knee = _paired_angle(rng, 142.0, 25.0, 65.0, 180.0)
        left_hip, right_hip = _paired_angle(rng, 146.0, 24.0, 65.0, 180.0)
        horizontal_score = _bounded_normal(rng, 0.82, 0.12, 0.58, 1.0)
        values = {
            "bbox_aspect_ratio": _bounded_normal(rng, 1.65, 0.38, 0.95, 2.65),
            "vertical_span_norm": _bounded_normal(rng, 1.22, 0.35, 0.45, 2.05),
            "torso_tilt_deg": _bounded_normal(rng, 75.0, 14.0, 42.0, 90.0),
            "horizontal_score": horizontal_score,
            "movement_norm": _bounded_normal(rng, 0.012, 0.018, 0.0, 0.08),
            "recent_drop_norm": _bounded_normal(rng, 0.02, 0.08, -0.16, 0.26),
            "vertical_velocity_norm_s": _bounded_normal(rng, 0.035, 0.08, 0.0, 0.34),
            "hip_y": _bounded_normal(rng, 0.68, 0.10, 0.38, 0.93),
            "ankle_y": _bounded_normal(rng, 0.72, 0.13, 0.36, 0.99),
            "body_center_y": _bounded_normal(rng, 0.63, 0.11, 0.34, 0.92),
        }
    elif label == "FALLING":
        left_knee, right_knee = _paired_angle(rng, 142.0, 26.0, 50.0, 180.0)
        left_hip, right_hip = _paired_angle(rng, 138.0, 26.0, 50.0, 180.0)
        values = {
            "bbox_aspect_ratio": _bounded_normal(rng, 1.05, 0.35, 0.48, 1.95),
            "vertical_span_norm": _bounded_normal(rng, 1.95, 0.50, 0.85, 3.10),
            "torso_tilt_deg": _bounded_normal(rng, 48.0, 20.0, 8.0, 90.0),
            "horizontal_score": _bounded_normal(rng, 0.50, 0.22, 0.12, 0.94),
            "movement_norm": _bounded_normal(rng, 0.18, 0.08, 0.06, 0.50),
            "recent_drop_norm": _bounded_normal(rng, 0.68, 0.22, 0.28, 1.35),
            "vertical_velocity_norm_s": _bounded_normal(rng, 1.55, 0.55, 0.58, 3.10),
            "hip_y": _bounded_normal(rng, 0.62, 0.13, 0.32, 0.95),
            "ankle_y": _bounded_normal(rng, 0.78, 0.12, 0.42, 0.99),
            "body_center_y": _bounded_normal(rng, 0.56, 0.13, 0.27, 0.91),
        }
    elif label == "COLLAPSING":
        left_knee, right_knee = _paired_angle(rng, 132.0, 30.0, 45.0, 180.0)
        left_hip, right_hip = _paired_angle(rng, 132.0, 30.0, 45.0, 180.0)
        values = {
            "bbox_aspect_ratio": _bounded_normal(rng, 1.42, 0.35, 0.82, 2.35),
            "vertical_span_norm": _bounded_normal(rng, 1.45, 0.42, 0.55, 2.45),
            "torso_tilt_deg": _bounded_normal(rng, 65.0, 18.0, 28.0, 90.0),
            "horizontal_score": _bounded_normal(rng, 0.70, 0.18, 0.34, 1.0),
            "movement_norm": _bounded_normal(rng, 0.10, 0.06, 0.025, 0.32),
            "recent_drop_norm": _bounded_normal(rng, 0.58, 0.22, 0.22, 1.25),
            "vertical_velocity_norm_s": _bounded_normal(rng, 1.10, 0.45, 0.35, 2.55),
            "hip_y": _bounded_normal(rng, 0.70, 0.12, 0.36, 0.96),
            "ankle_y": _bounded_normal(rng, 0.76, 0.13, 0.38, 0.99),
            "body_center_y": _bounded_normal(rng, 0.66, 0.12, 0.34, 0.95),
        }
    else:  # pragma: no cover - guarded by normalize_label
        raise ValueError(label)

    result = {
        "pose_quality": pose_quality,
        "visible_fraction": visible_fraction,
        "torso_length": torso_length,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "left_knee_angle": left_knee,
        "right_knee_angle": right_knee,
        "left_hip_angle": left_hip,
        "right_hip_angle": right_hip,
        "head_hip_dx_norm": head_hip_dx_norm,
        "wrist_height_diff_norm": wrist_height_diff_norm,
        "depth_spread_norm": depth_spread_norm,
        **values,
    }
    return {name: float(np.clip(result.get(name, 0.0), -5.0, 180.0)) for name in FEATURE_COLUMNS}


def generate_synthetic_samples(
    dataset_path: Path = DEFAULT_DATASET,
    samples_per_label: int = 160,
    sessions_per_label: int = 4,
    seed: int = 42,
    append: bool = False,
) -> pd.DataFrame:
    """Create a deterministic bootstrap dataset for local model smoke testing.

    These rows are mathematically plausible feature vectors, not clinical evidence.
    Use them to exercise the scikit-learn pipeline before replacing/augmenting them
    with real MediaPipe captures from the Raspberry Pi camera.
    """
    if samples_per_label < 5:
        raise ValueError("samples_per_label must be at least 5")
    if sessions_per_label < 2:
        raise ValueError("sessions_per_label must be at least 2 for session-aware evaluation")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    labels = ["STANDING", "SITTING", "LYING", "FALLING", "COLLAPSING"]
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for label in labels:
        for index in range(samples_per_label):
            session_number = index % sessions_per_label
            session_rng = np.random.default_rng(seed + labels.index(label) * 100 + session_number)
            session_bias = {
                "torso_length": float(np.clip(session_rng.normal(0.235, 0.035), 0.14, 0.34)),
            }
            features = _synthetic_feature_row(label, rng, session_bias)
            rows.append(
                {
                    "sample_id": f"synthetic-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{seed}-{label}-{index}')}",
                    "session_id": f"synthetic-{label.lower()}-{session_number + 1:02d}",
                    "captured_at": (start_time + timedelta(seconds=len(rows))).isoformat(),
                    "label": label,
                    "feature_version": FEATURE_VERSION,
                    "camera": "synthetic-bootstrap",
                    **features,
                }
            )

    frame = pd.DataFrame(rows, columns=METADATA_COLUMNS + FEATURE_COLUMNS)
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if append and dataset_path.exists():
        existing = pd.read_csv(dataset_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(dataset_path, index=False)
    return frame


def load_dataset(path: Path = DEFAULT_DATASET) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + ["label", "session_id", "feature_version"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    frame = frame[frame["feature_version"].astype(str) == FEATURE_VERSION].copy()
    frame["label"] = frame["label"].map(normalize_label)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["label"])
    if frame.empty:
        raise ValueError("No valid rows remain after feature-version and missing-data checks")
    return frame


def _session_stratified_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    sessions_by_label = frame.groupby("label")["session_id"].unique().to_dict()
    if not sessions_by_label or any(len(sessions) < 2 for sessions in sessions_by_label.values()):
        return None
    rng = np.random.default_rng(42)
    test_sessions: set[str] = set()
    for label, sessions in sessions_by_label.items():
        sessions = np.array(sorted(map(str, sessions)))
        rng.shuffle(sessions)
        count = max(1, int(round(len(sessions) * 0.25)))
        test_sessions.update(sessions[:count].tolist())
    mask = frame["session_id"].astype(str).isin(test_sessions).to_numpy()
    train_idx = np.flatnonzero(~mask)
    test_idx = np.flatnonzero(mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    if set(frame.iloc[train_idx]["label"]) != set(frame["label"]) or set(frame.iloc[test_idx]["label"]) != set(frame["label"]):
        return None
    return train_idx, test_idx


def _split(frame: pd.DataFrame):
    labels = frame["label"].to_numpy()
    session_split = _session_stratified_split(frame)
    if session_split is not None:
        train_idx, test_idx = session_split
        return train_idx, test_idx, "session-stratified"
    if min(Counter(labels).values()) < 2:
        return np.arange(len(frame)), np.array([], dtype=int), "train-only"
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(frame, labels))
    return train_idx, test_idx, "stratified-row"


def train_from_dataset(dataset_path: Path = DEFAULT_DATASET, model_path: Path = DEFAULT_MODEL) -> dict[str, Any]:
    frame = load_dataset(Path(dataset_path))
    counts = Counter(frame["label"])
    missing_classes = ALLOWED_LABELS - set(counts)
    if missing_classes:
        LOGGER.warning("Dataset does not contain these classes: %s", sorted(missing_classes))
    if min(counts.values()) < 30:
        LOGGER.warning("Some classes have fewer than 30 samples; collect more data before trusting the model")

    X = frame[FEATURE_COLUMNS].astype(float).to_numpy()
    y = frame["label"].astype(str).to_numpy()
    train_idx, test_idx, split_strategy = _split(frame)

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=16,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        oob_score=len(train_idx) >= 100,
    )
    model.fit(X[train_idx], y[train_idx])

    report: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,
        "rows": len(frame),
        "classes": sorted(model.classes_.tolist()),
        "class_counts": dict(counts),
        "split_strategy": split_strategy,
        "data_warning": (
            "Synthetic/bootstrap rows are useful for pipeline testing only. "
            "Validate with real unseen people, lighting, clothing, distances, and camera positions before trusting the model."
        ),
    }
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        report["top_feature_importances"] = [
            {"feature": name, "importance": round(float(value), 5)} for name, value in importances[:10]
        ]
    if len(test_idx):
        predictions = model.predict(X[test_idx])
        probabilities = model.predict_proba(X[test_idx])
        max_probabilities = probabilities.max(axis=1)
        report["classification_report"] = classification_report(y[test_idx], predictions, output_dict=True, zero_division=0)
        report["confusion_matrix_labels"] = model.classes_.tolist()
        report["confusion_matrix"] = confusion_matrix(y[test_idx], predictions, labels=model.classes_).tolist()
        report["mean_prediction_confidence"] = float(np.mean(max_probabilities))
        report["low_confidence_fraction_below_0_66"] = float(np.mean(max_probabilities < 0.66))
    else:
        report["evaluation"] = "Insufficient independent data for a holdout set. Model was trained without validation."

    bundle = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "feature_version": FEATURE_VERSION,
        "trained_at": report["trained_at"],
        "minimum_confidence": 0.66,
        "labels": sorted(model.classes_.tolist()),
        "report": report,
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path = model_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect, synthesize, and train confidence-aware CareAgent pose data")
    parser.add_argument("--collect", metavar="LABEL")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--generate-synthetic", action="store_true", help="Create a deterministic bootstrap pose_dataset.csv")
    parser.add_argument("--synthetic-per-label", type=int, default=160)
    parser.add_argument("--synthetic-sessions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true", help="Append synthetic samples to an existing dataset instead of replacing it")
    args = parser.parse_args()
    if args.collect:
        collect_samples(args.collect, args.samples, args.dataset)
    if args.generate_synthetic:
        frame = generate_synthetic_samples(
            args.dataset,
            samples_per_label=args.synthetic_per_label,
            sessions_per_label=args.synthetic_sessions,
            seed=args.seed,
            append=args.append,
        )
        print(f"Saved synthetic bootstrap dataset: {args.dataset} ({len(frame)} rows)")
    if args.train:
        train_from_dataset(args.dataset, args.model)
    if not (args.collect or args.generate_synthetic or args.train):
        parser.print_help()


if __name__ == "__main__":
    main()
