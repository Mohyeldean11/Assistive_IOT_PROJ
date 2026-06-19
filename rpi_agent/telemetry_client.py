from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

LOGGER = logging.getLogger("careagent.telemetry")


class TelemetryClient:
    """Reliable local-network uploader with disk-backed retry spool."""

    def __init__(self, url: str, api_key: str, spool_dir: Path, timeout: float = 12.0):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key, "Accept": "application/json"})

    def _post(self, payload: dict[str, Any]) -> bool:
        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        if response.status_code in (200, 201):
            return True
        if 400 <= response.status_code < 500:
            LOGGER.error("CareAgent rejected payload (%s): %s", response.status_code, response.text[:300])
            return False
        response.raise_for_status()
        return False

    def _spool_path(self, payload: dict[str, Any]) -> Path:
        event_id = str(payload.get("event_id", int(time.time() * 1000))).replace("/", "_")
        return self.spool_dir / f"{event_id}.json"

    def queue(self, payload: dict[str, Any]) -> Path:
        target = self._spool_path(payload)
        fd, temp_name = tempfile.mkstemp(prefix="pending_", suffix=".json", dir=self.spool_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return target

    def send(self, payload: dict[str, Any]) -> bool:
        try:
            if self._post(payload):
                return True
        except requests.RequestException as exc:
            LOGGER.warning("CareAgent upload failed: %s", exc)
        self.queue(payload)
        return False

    def flush_pending(self, limit: int = 50) -> tuple[int, int]:
        sent = 0
        failed = 0
        for path in sorted(self.spool_dir.glob("*.json"))[:limit]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if self._post(payload):
                    path.unlink(missing_ok=True)
                    sent += 1
                else:
                    failed += 1
            except (OSError, json.JSONDecodeError, requests.RequestException) as exc:
                LOGGER.warning("Pending upload %s failed: %s", path.name, exc)
                failed += 1
                break
        return sent, failed
