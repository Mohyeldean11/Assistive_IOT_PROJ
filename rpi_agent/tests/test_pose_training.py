from collections import Counter
from pathlib import Path

from pose_features import FEATURE_COLUMNS, FEATURE_VERSION
from pose_training import ALLOWED_LABELS, generate_synthetic_samples, load_dataset


def test_synthetic_samples_cover_all_pose_classes(tmp_path: Path):
    dataset = tmp_path / "synthetic.csv"
    frame = generate_synthetic_samples(dataset, samples_per_label=6, sessions_per_label=2, seed=7)
    assert dataset.exists()
    assert len(frame) == 6 * len(ALLOWED_LABELS)
    assert set(frame["label"]) == ALLOWED_LABELS
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert set(frame["feature_version"]) == {FEATURE_VERSION}

    loaded = load_dataset(dataset)
    counts = Counter(loaded["label"])
    assert all(counts[label] == 6 for label in ALLOWED_LABELS)
