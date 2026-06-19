from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from camera_pipeline import CameraPosePipeline
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


def _split(frame: pd.DataFrame):
    labels = frame["label"].to_numpy()
    unique_sessions = frame["session_id"].nunique()
    if unique_sessions >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(frame, labels, groups=frame["session_id"]))
        split_strategy = "session-grouped"
    else:
        if min(Counter(labels).values()) < 2:
            return np.arange(len(frame)), np.array([], dtype=int), "train-only"
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(frame, labels))
        split_strategy = "stratified-row"
    return train_idx, test_idx, split_strategy


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
        "warning": "Prototype pose classifier; validate on unseen people, lighting, clothing, distances, and camera positions.",
    }
    if len(test_idx):
        predictions = model.predict(X[test_idx])
        probabilities = model.predict_proba(X[test_idx])
        max_probabilities = probabilities.max(axis=1)
        report["classification_report"] = classification_report(y[test_idx], predictions, output_dict=True, zero_division=0)
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
    parser = argparse.ArgumentParser(description="Collect and train confidence-aware CareAgent pose data")
    parser.add_argument("--collect", metavar="LABEL")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.collect:
        collect_samples(args.collect, args.samples, args.dataset)
    elif args.train:
        train_from_dataset(args.dataset, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
