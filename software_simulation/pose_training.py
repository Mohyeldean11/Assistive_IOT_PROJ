import csv
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import Mediapipe_Class

FEATURE_COLUMNS = [
    'shoulder_distance',
    'hip_distance',
    'knee_distance',
    'vertical_span',
    'spine_drop',
    'is_horizontal',
    'depth_spread',
    'movement_score',
    'recent_drop',
    'shoulder_hip_height',
    'torso_angle',
    'left_leg_angle',
    'right_leg_angle',
    'head_hip_ratio',
    'shoulder_knee_ratio',
    'hip_knee_ratio',
    'head_slope',
]

DEFAULT_DATASET_PATH = Path('pose_dataset.csv')
DEFAULT_MODEL_PATH = Path('pose_model.pkl')


def vectorize_features(features: dict) -> list[float]:
    return [
        float(features.get(name, 0.0)) if name != 'is_horizontal' else float(int(features.get(name, False)))
        for name in FEATURE_COLUMNS
    ]


def load_model(model_path: Optional[Path] = None):
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        return None
    return joblib.load(path)


def save_model(model, model_path: Optional[Path] = None):
    path = Path(model_path or DEFAULT_MODEL_PATH)
    joblib.dump(model, path)
    return path


def load_dataset(dataset_path: Optional[Path] = None):
    path = Path(dataset_path or DEFAULT_DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS + ['label'] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    y = df['label'].astype(str).to_numpy()
    return X, y, df


def save_training_sample(pose_sequence: list[dict], label: str, dataset_path: Optional[Path] = None):
    path = Path(dataset_path or DEFAULT_DATASET_PATH)
    features = Mediapipe_Class.frame_classifier().compute_spatial_features(pose_sequence)
    row = dict(zip(FEATURE_COLUMNS, vectorize_features(features)))
    row['label'] = str(label)

    header = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FEATURE_COLUMNS + ['label'])
        if header:
            writer.writeheader()
        writer.writerow(row)
    return row


def train_classifier(X: np.ndarray, y: np.ndarray):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def train_from_dataset(dataset_path: Optional[Path] = None, model_path: Optional[Path] = None):
    X, y, df = load_dataset(dataset_path)
    model = train_classifier(X, y)
    save_model(model, model_path)
    return model, df


def predict_from_features(features: dict, model):
    if model is None or not features:
        return None
    vector = np.array([vectorize_features(features)])
    return model.predict(vector)[0]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pose model training and dataset collection')
    parser.add_argument('--collect', type=str, help='Label name to collect one pose sample')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to collect')
    parser.add_argument('--dataset', type=str, default=str(DEFAULT_DATASET_PATH), help='CSV dataset path')
    parser.add_argument('--train', action='store_true', help='Train model from dataset CSV')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL_PATH), help='Path to save trained model')
    args = parser.parse_args()

    if args.collect:
        classifier = Mediapipe_Class.frame_classifier()
        for i in range(args.samples):
            print(f'Collecting sample {i + 1}/{args.samples} for label {args.collect}')
            sequence = classifier.get_sequence(count=4)
            sample = save_training_sample(sequence, args.collect, args.dataset)
            print('Saved sample:', sample)
        print('Collection complete. Run with --train to create a model.')
    elif args.train:
        print('Training from dataset:', args.dataset)
        model, df = train_from_dataset(args.dataset, args.model)
        print('Training complete. Saved model to', args.model)
        print('Classes:', model.classes_)
    else:
        parser.print_help()
