from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Settings

LOGGER = logging.getLogger("careagent.camera")
POSE_LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
    "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
    "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]


class CameraPosePipeline:
    """Persistent C930/C930e + MediaPipe VIDEO-mode pipeline.

    Keeping VideoCapture and PoseLandmarker open avoids auto-exposure resets and lets
    MediaPipe tracking stabilize landmarks between frames.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.capture: cv2.VideoCapture | None = None
        self.landmarker: Any = None
        self.mp: Any = None
        self._last_timestamp_ms = 0
        self.last_frame: np.ndarray | None = None

    def open(self) -> None:
        if not Path(self.settings.pose_model_path).exists():
            raise FileNotFoundError(
                f"Pose landmarker model not found: {self.settings.pose_model_path}. "
                "Place pose_landmarker_lite.task in rpi_agent/models/."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed in the Python 3.11 environment") from exc
        self.mp = mp

        capture = cv2.VideoCapture(self.settings.camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.settings.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"USB camera index {self.settings.camera_index} is not available")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
        capture.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.settings.pose_model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.settings.min_detection_confidence,
            min_pose_presence_confidence=self.settings.min_presence_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

        for _ in range(self.settings.camera_warmup_frames):
            self.capture.read()
        LOGGER.info(
            "Camera opened at requested %sx%s @ %s fps",
            self.settings.camera_width,
            self.settings.camera_height,
            self.settings.camera_fps,
        )

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    @staticmethod
    def frame_quality(frame: np.ndarray) -> dict[str, float | bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        clipped_fraction = float(np.mean((gray < 5) | (gray > 250)))
        usable = 28 <= brightness <= 225 and blur_variance >= 22 and clipped_fraction <= 0.45
        quality = min(1.0, blur_variance / 120.0) * max(0.0, 1.0 - abs(brightness - 125.0) / 125.0) * (1.0 - clipped_fraction)
        return {"brightness": brightness, "blur_variance": blur_variance, "clipped_fraction": clipped_fraction, "usable": usable, "score": quality}

    def _timestamp_ms(self) -> int:
        timestamp = time.monotonic_ns() // 1_000_000
        if timestamp <= self._last_timestamp_ms:
            timestamp = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp
        return timestamp

    @staticmethod
    def _extract_landmarks(result: Any) -> dict[str, dict[str, float]]:
        if not result.pose_landmarks:
            return {}
        landmarks: dict[str, dict[str, float]] = {}
        for index, point in enumerate(result.pose_landmarks[0]):
            # MediaPipe documents ``presence`` as optional/unset depending on the
            # model/build. Treat an unset or zero presence value as unavailable
            # rather than rejecting an otherwise visible landmark.
            visibility_value = getattr(point, "visibility", None)
            visibility = 1.0 if visibility_value is None else float(visibility_value)
            presence_value = getattr(point, "presence", None)
            try:
                presence = float(presence_value) if presence_value is not None else visibility
            except (TypeError, ValueError):
                presence = visibility
            if presence <= 0.0 and visibility > 0.0:
                presence = visibility
            landmarks[POSE_LANDMARK_NAMES[index]] = {
                "x": float(point.x),
                "y": float(point.y),
                "z": float(point.z),
                "visibility": visibility,
                "presence": presence,
            }
        return landmarks


    def encode_latest_snapshot(self, jpeg_quality: int = 70, max_width: int = 960) -> dict[str, Any] | None:
        """Return the latest camera frame as base64 JPEG for the dashboard.

        The image is resized before encoding to keep the JSON request small enough
        for local-network transmission and mobile dashboard refreshes.
        """
        if self.last_frame is None:
            return None
        frame = self.last_frame
        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / float(frame.shape[1])
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        quality = int(max(35, min(95, jpeg_quality)))
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return {
            "mime_type": "image/jpeg",
            "encoding": "base64",
            "image_base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }

    def read_observation(self) -> dict[str, Any]:
        if self.capture is None or self.landmarker is None or self.mp is None:
            raise RuntimeError("CameraPosePipeline.open() must be called first")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("USB camera frame capture failed")
        self.last_frame = frame.copy()
        quality = self.frame_quality(frame)
        timestamp_ms = self._timestamp_ms()
        # Do not block MediaPipe inference because of a heuristic brightness/blur
        # check. The original regression happened here: valid frames were returned
        # with empty landmarks before MediaPipe was even called. Quality remains a
        # diagnostic signal and can reduce confidence later, but inference always runs.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        landmarks = self._extract_landmarks(result)
        if self.settings.save_pose_debug_log and not landmarks:
            LOGGER.warning("No MediaPipe pose landmarks detected. quality=%s brightness=%.1f blur=%.1f clipped=%.2f", quality, quality.get("brightness", 0.0), quality.get("blur_variance", 0.0), quality.get("clipped_fraction", 0.0))
        if self.settings.save_debug_frames:
            debug_dir = Path(__file__).resolve().parent / "logs" / "debug_frames"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"frame_{timestamp_ms}.jpg"), frame)
        return {
            "timestamp_ms": timestamp_ms,
            "landmarks": landmarks,
            "landmark_count": len(landmarks),
            "frame_quality": quality,
        }
