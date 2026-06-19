"""Small Raspberry Pi camera/pose diagnostic for CareAgent AI.

Run from rpi_agent with the Python 3.11 environment active:
    python pose_diagnostic.py --frames 30
"""
from __future__ import annotations

import argparse
import time
from collections import deque

from camera_pipeline import CameraPosePipeline
from config import settings
from pose_classifier import PoseClassifier
from pose_features import compute_features, compute_legacy_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    classifier = PoseClassifier(
        settings.pose_classifier_path,
        settings.classifier_min_confidence,
        settings.floor_y_normalized,
        settings.legacy_pose_classifier_path,
    )
    observations: deque[dict] = deque(maxlen=10)
    with CameraPosePipeline(settings) as camera:
        for index in range(args.frames):
            observation = camera.read_observation()
            observations.append(observation)
            window = list(observations)
            features = compute_features(window, settings.min_landmark_visibility)
            legacy_features = compute_legacy_features(window)
            result = classifier.classify(features, legacy_features)
            quality = observation.get("frame_quality", {})
            print(
                f"{index + 1:02d} landmarks={len(observation.get('landmarks', {})):02d} "
                f"frame_usable={quality.get('usable')} features={'yes' if features else 'no '} "
                f"visible={features.get('visible_fraction', 0):.2f} quality={features.get('pose_quality', 0):.2f} "
                f"raw={result['raw_label']:<10} stable={result['label']:<10} "
                f"source={result['source']} confidence={result['confidence']:.2f}"
            )
            time.sleep(max(0.05, settings.analysis_interval_seconds))


if __name__ == "__main__":
    main()
