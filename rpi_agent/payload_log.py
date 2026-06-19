from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class JsonEventLog:
    """Append-only JSON Lines log; one valid JSON object per line."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, payload: dict[str, Any]) -> Path:
        target = self.directory / f"readings_{datetime.now().strftime('%Y_%m_%d')}.jsonl"
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock, target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return target
