from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Iterable

from config import BASE_DIR, settings
from telemetry_client import TelemetryClient

LOGGER = logging.getLogger("careagent.forwarder")
STATE_FILE = BASE_DIR / "spool" / "forwarder_state.json"


def iter_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    yield index, parsed
        return
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, dict):
        yield 0, parsed
    elif isinstance(parsed, list):
        for index, record in enumerate(parsed):
            if isinstance(record, dict):
                yield index, record


def load_state() -> dict[str, int]:
    try:
        parsed = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, int]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE_FILE)


def deterministic_event_id(path: Path, index: int, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"careagent:{path.resolve()}:{index}:{digest}"))


def forward_file(path: Path) -> tuple[int, int]:
    path = path.resolve()
    client = TelemetryClient(settings.careagent_url, settings.device_api_key, BASE_DIR / "spool", settings.request_timeout_seconds)
    state = load_state()
    state_key = str(path)
    last_sent = int(state.get(state_key, -1))
    sent = 0
    failed = 0
    for index, payload in iter_records(path):
        if index <= last_sent:
            continue
        payload.setdefault("event_id", deterministic_event_id(path, index, payload))
        if client.send(payload):
            state[state_key] = index
            save_state(state)
            sent += 1
        else:
            failed += 1
            break
    return sent, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward legacy JSON/JSONL logs to the local CareAgent AI web app")
    parser.add_argument("path", type=Path, help="JSON array file, JSON object file, or JSONL file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    sent, failed = forward_file(args.path)
    print(f"Forwarded: {sent}; failed/queued: {failed}")


if __name__ == "__main__":
    main()
