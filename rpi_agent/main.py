from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_module import OptionalLocalAI
from camera_pipeline import CameraPosePipeline
from config import BASE_DIR, settings
from emergency_engine import EmergencyEngine
from health_status import deterministic_health_status
from payload_log import JsonEventLog
from pose_classifier import PoseClassifier, normalize_pose_label
from pose_features import compute_features, compute_legacy_features
from sensors import SensorReader
from stroke_detector import StrokeRiskEngine
from telemetry_client import TelemetryClient


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_payload(
    sensor_data: dict[str, Any],
    pose_result: dict[str, Any],
    risk: dict[str, Any],
    frame_quality: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "deviceId": settings.device_id,
        "captured_at": utc_now(),
        **sensor_data,
        "pose": normalize_pose_label(pose_result.get("label")),
        "pose_confidence": pose_result.get("confidence", 0.0),
        "pose_diagnostics": {
            "raw_label": pose_result.get("raw_label"),
            "raw_confidence": pose_result.get("raw_confidence"),
            "source": pose_result.get("source"),
            "model_candidate": pose_result.get("model_candidate"),
            "model_confidence": pose_result.get("model_confidence"),
            "rule_candidate": pose_result.get("rule_candidate"),
            "rule_confidence": pose_result.get("rule_confidence"),
            "frame_quality": frame_quality,
        },
        "stroke_risk": risk,
    }
    if snapshot:
        payload["snapshot"] = snapshot
    decision = EmergencyEngine.evaluate(payload)
    payload["environment_status"] = decision.environment_status
    payload["emergency"] = decision.emergency
    payload["emergency_reasons"] = decision.reasons
    payload["health_status"] = deterministic_health_status(payload)
    return payload


def run_agent(run_once: bool = False) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("careagent.main")

    sensor_reader = SensorReader(simulate=settings.simulate_sensors)
    uploader = TelemetryClient(
        settings.careagent_url,
        settings.device_api_key,
        BASE_DIR / "spool",
        settings.request_timeout_seconds,
    )
    event_log = JsonEventLog(BASE_DIR / "logs")
    pose_classifier = PoseClassifier(
        settings.pose_classifier_path,
        settings.classifier_min_confidence,
        settings.floor_y_normalized,
        settings.legacy_pose_classifier_path,
    )
    risk_engine = StrokeRiskEngine(settings.min_landmark_visibility)
    local_ai = OptionalLocalAI(settings.enable_local_llm, settings.ollama_model)
    observations: deque[dict[str, Any]] = deque(maxlen=10)

    last_periodic_upload = 0.0
    last_emergency_upload = 0.0
    last_sensor_read = 0.0
    last_sensor_data: dict[str, Any] | None = None
    last_payload: dict[str, Any] | None = None
    flushed, failed = uploader.flush_pending(limit=50)
    logger.info("Startup pending queue: flushed=%s failed=%s", flushed, failed)

    with CameraPosePipeline(settings) as camera:
        while True:
            cycle_start = time.monotonic()
            try:
                observation = camera.read_observation()
                observations.append(observation)
                observation_window = list(observations)
                features = compute_features(observation_window, settings.min_landmark_visibility)
                legacy_features = compute_legacy_features(observation_window)
                if settings.save_pose_debug_log and not features:
                    logger.warning(
                        "Pose features empty. landmark_count=%s quality=%s",
                        observation.get("landmark_count", len(observation.get("landmarks", {}))),
                        observation.get("frame_quality"),
                    )
                pose_result = pose_classifier.classify(features, legacy_features)
                risk = risk_engine.update(observation, pose_result["label"], pose_result["confidence"])
                now = time.monotonic()
                if last_sensor_data is None or now - last_sensor_read >= settings.sensor_read_interval_seconds:
                    last_sensor_data = sensor_reader.read()
                    last_sensor_read = now
                frame_quality = observation.get("frame_quality", {})
                last_payload = build_payload(last_sensor_data, pose_result, risk, frame_quality)
                ai_summary = local_ai.summarize(last_payload)
                if ai_summary:
                    last_payload["ai_summary"] = ai_summary

                now = time.monotonic()
                periodic_due = now - last_periodic_upload >= settings.upload_interval_seconds
                emergency_due = (
                    last_payload["emergency"]
                    and now - last_emergency_upload >= settings.emergency_upload_cooldown_seconds
                )
                if periodic_due or emergency_due or run_once:
                    payload_to_upload = dict(last_payload)
                    if settings.enable_snapshot_upload:
                        snapshot = camera.encode_latest_snapshot(settings.snapshot_jpeg_quality, settings.snapshot_max_width)
                        if snapshot:
                            payload_to_upload["snapshot"] = snapshot
                    log_payload = dict(payload_to_upload)
                    if isinstance(log_payload.get("snapshot"), dict):
                        log_snapshot = dict(log_payload["snapshot"])
                        log_snapshot.pop("image_base64", None)
                        log_payload["snapshot"] = log_snapshot
                    event_log.append(log_payload)
                    sent = uploader.send(payload_to_upload)
                    flushed, failed = uploader.flush_pending(limit=20) if sent else (0, 0)
                    logger.info(
                        "Telemetry %s | pose=%s %.2f risk=%s emergency=%s pending_flushed=%s failed=%s",
                        "sent" if sent else "queued",
                        payload_to_upload["pose"],
                        payload_to_upload["pose_confidence"],
                        payload_to_upload["stroke_risk"]["risk_level"],
                        payload_to_upload["emergency"],
                        flushed,
                        failed,
                    )
                    if periodic_due or run_once:
                        last_periodic_upload = now
                    if emergency_due:
                        last_emergency_upload = now
                    if run_once:
                        return
            except KeyboardInterrupt:
                logger.info("Stopping CareAgent Raspberry Pi agent")
                return
            except Exception:
                logger.exception("Analysis cycle failed")

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.05, settings.analysis_interval_seconds - elapsed))


def main() -> None:
    parser = argparse.ArgumentParser(description="CareAgent AI Raspberry Pi 5 monitoring agent")
    parser.add_argument("--once", action="store_true", help="Analyze and upload one reading, then exit")
    parser.add_argument("--collect", metavar="LABEL", help="Collect pose samples; delegates to pose_training.py")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--train", action="store_true", help="Train the pose classifier from pose_dataset.csv")
    parser.add_argument("--generate-synthetic", action="store_true", help="Create a bootstrap synthetic pose_dataset.csv")
    parser.add_argument("--synthetic-per-label", type=int, default=160)
    parser.add_argument("--synthetic-sessions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.collect or args.generate_synthetic or args.train:
        import pose_training
        if args.collect:
            pose_training.collect_samples(args.collect, args.samples)
        if args.generate_synthetic:
            frame = pose_training.generate_synthetic_samples(
                samples_per_label=args.synthetic_per_label,
                sessions_per_label=args.synthetic_sessions,
                seed=args.seed,
            )
            print(f"Saved synthetic bootstrap dataset: {pose_training.DEFAULT_DATASET} ({len(frame)} rows)")
        if args.train:
            pose_training.train_from_dataset()
        return
    run_agent(run_once=args.once)


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(f"This Raspberry Pi agent is pinned to Python 3.11; detected {sys.version.split()[0]}")
    main()
