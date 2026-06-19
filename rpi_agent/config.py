from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    device_id: str = os.getenv("DEVICE_ID", "rpi-01")
    careagent_url: str = os.getenv("CAREAGENT_URL", "http://127.0.0.1:5000/api/readings")
    device_api_key: str = os.getenv("DEVICE_API_KEY", "replace-with-a-device-api-key")
    upload_interval_seconds: int = int(os.getenv("UPLOAD_INTERVAL_SECONDS", "120"))
    sensor_read_interval_seconds: float = float(os.getenv("SENSOR_READ_INTERVAL_SECONDS", "5.0"))
    analysis_interval_seconds: float = float(os.getenv("ANALYSIS_INTERVAL_SECONDS", "1.0"))
    emergency_upload_cooldown_seconds: int = int(os.getenv("EMERGENCY_UPLOAD_COOLDOWN_SECONDS", "60"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))

    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_width: int = int(os.getenv("CAMERA_WIDTH", "1280"))
    camera_height: int = int(os.getenv("CAMERA_HEIGHT", "720"))
    camera_fps: int = int(os.getenv("CAMERA_FPS", "30"))
    camera_warmup_frames: int = int(os.getenv("CAMERA_WARMUP_FRAMES", "12"))
    pose_model_path: Path = Path(os.getenv("POSE_LANDMARKER_MODEL", str(BASE_DIR / "models" / "pose_landmarker_lite.task")))
    pose_classifier_path: Path = Path(os.getenv("POSE_CLASSIFIER_MODEL", str(BASE_DIR / "models" / "pose_classifier.joblib")))
    legacy_pose_classifier_path: Path = Path(os.getenv("LEGACY_POSE_CLASSIFIER_MODEL", str(BASE_DIR / "pose_model.pkl")))
    min_detection_confidence: float = float(os.getenv("MIN_POSE_DETECTION_CONFIDENCE", "0.50"))
    min_presence_confidence: float = float(os.getenv("MIN_POSE_PRESENCE_CONFIDENCE", "0.50"))
    min_tracking_confidence: float = float(os.getenv("MIN_TRACKING_CONFIDENCE", "0.50"))
    min_landmark_visibility: float = float(os.getenv("MIN_LANDMARK_VISIBILITY", "0.40"))
    min_pose_quality: float = float(os.getenv("MIN_POSE_QUALITY", "0.35"))
    classifier_min_confidence: float = float(os.getenv("CLASSIFIER_MIN_CONFIDENCE", "0.60"))

    simulate_sensors: bool = _bool("SIMULATE_SENSORS", True)
    enable_local_llm: bool = _bool("ENABLE_LOCAL_LLM", False)
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    save_debug_frames: bool = _bool("SAVE_DEBUG_FRAMES", False)
    save_pose_debug_log: bool = _bool("SAVE_POSE_DEBUG_LOG", True)
    enable_snapshot_upload: bool = _bool("ENABLE_SNAPSHOT_UPLOAD", True)
    snapshot_jpeg_quality: int = int(os.getenv("SNAPSHOT_JPEG_QUALITY", "70"))
    snapshot_max_width: int = int(os.getenv("SNAPSHOT_MAX_WIDTH", "960"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Optional scene calibration. Set after measuring the camera view; leave blank/0 to disable.
    floor_y_normalized: float = float(os.getenv("FLOOR_Y_NORMALIZED", "0"))


settings = Settings()
