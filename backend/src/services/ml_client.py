"""Async HTTP client for the ML model service."""

import logging
from pathlib import Path

import httpx

from config.settings import get_settings

logger = logging.getLogger("mindscope")
settings = get_settings()


class MLClient:
    """Communicates with the standalone ML model API."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or settings.ML_MODEL_URL).rstrip("/")
        self.timeout = timeout

    async def predict_extended(self, audio_path: str, participant_id: str = "unknown") -> dict:
        """POST multipart to /predict/extended and return parsed JSON."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(audio_path, "rb") as f:
                filename = Path(audio_path).name
                resp = await client.post(
                    f"{self.base_url}/predict/extended",
                    files={"file": (filename, f, "application/octet-stream")},
                    params={"participant_id": participant_id},
                )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = None
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text
                message = detail or str(exc)
                raise RuntimeError(
                    f"Model API error ({resp.status_code}): {message}"
                ) from exc
            return resp.json()

    async def health_check(self) -> dict:
        """GET /health from the ML model service."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
