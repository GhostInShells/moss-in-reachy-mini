import base64
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import json

logger = logging.getLogger(__name__)

__all__ = [
    "SSEEvent",
    "parse_sse_line",
    "extract_audio_chunk",
]


@dataclass
class SSEEvent:
    """Server-Sent Event from Xiaomi TTS API."""

    data: str = ""

    @property
    def is_done(self) -> bool:
        return self.data.strip() == "[DONE]"

    @property
    def is_empty(self) -> bool:
        return not self.data.strip()

    def parse_json(self) -> Optional[dict]:
        if self.is_done or self.is_empty:
            return None
        try:
            return json.loads(self.data)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Failed to parse SSE JSON: %s", self.data[:200])
            return None


def parse_sse_line(line: str) -> Optional[SSEEvent]:
    """Parse a single SSE line.

    SSE format:
        data: <json>

    Returns SSEEvent if the line is a data line, None otherwise.
    """
    line = line.strip()
    if not line:
        return None
    if line.startswith("data:"):
        data = line[len("data:"):].strip()
        return SSEEvent(data=data)
    return None


def extract_audio_chunk(event_json: dict) -> Optional[bytes]:
    """Extract PCM16 audio bytes from a Xiaomi TTS SSE response JSON.

    Response format (OpenAI-compatible):
    {
        "id": "...",
        "choices": [{
            "index": 0,
            "delta": {
                "audio": {
                    "data": "<base64_pcm16>"
                }
            },
            "finish_reason": null
        }]
    }

    Returns raw PCM16 bytes, or None if no audio data present.
    """
    try:
        choices = event_json.get("choices")
        if not choices:
            return None
        for choice in choices:
            delta = choice.get("delta")
            if not delta:
                continue
            audio = delta.get("audio")
            if not audio:
                continue
            data = audio.get("data")
            if not data:
                continue
            return base64.b64decode(data)
    except Exception:
        logger.debug("Failed to extract audio chunk", exc_info=True)
    return None


def pcm16_bytes_to_numpy(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw PCM16 bytes to numpy int16 array."""
    return np.frombuffer(pcm_bytes, dtype=np.int16)
