from __future__ import annotations

import json
import logging
from typing import Any

LOGGER = logging.getLogger("careagent.ai")


class OptionalLocalAI:
    """Optional local narrative generator.

    The LLM is deliberately excluded from pose classification, risk scoring, and
    emergency decisions. It can only summarize already-computed structured evidence.
    """

    def __init__(self, enabled: bool = False, model: str = "llama3.2:3b"):
        self.enabled = enabled
        self.model = model

    @staticmethod
    def build_prompt(payload: dict[str, Any]) -> str:
        evidence = {
            "pose": payload.get("pose"),
            "pose_confidence": payload.get("pose_confidence"),
            "stroke_risk": payload.get("stroke_risk"),
            "environment_status": payload.get("environment_status"),
            "emergency": payload.get("emergency"),
        }
        return (
            "You summarize an elderly-monitoring event for a caregiver. "
            "Use only the JSON evidence below. Do not diagnose stroke, invent symptoms, "
            "change the risk level, or recommend medication. Return strict JSON with keys "
            '"summary" and "recommended_action". The recommended action must be one of '
            '"continue monitoring", "check the person now", or "call local emergency services". '
            "Keep the summary under 35 words.\nEvidence:\n" + json.dumps(evidence, ensure_ascii=False)
        )

    def summarize(self, payload: dict[str, Any]) -> dict[str, str] | None:
        if not self.enabled:
            return None
        try:
            import ollama
            response = ollama.generate(model=self.model, prompt=self.build_prompt(payload), format="json")
            parsed = json.loads(response.response)
            action = parsed.get("recommended_action")
            if action not in {"continue monitoring", "check the person now", "call local emergency services"}:
                raise ValueError("Unsupported recommended_action")
            return {"summary": str(parsed.get("summary", ""))[:240], "recommended_action": action}
        except Exception:
            LOGGER.exception("Optional local AI summary failed")
            return None
