"""Request metrics middleware — batched DB writes to minimise per-request latency."""

import asyncio
import time
import logging
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from database import async_session_factory
from src.models import RequestMetric

logger = logging.getLogger("mindscope")

# Paths that should NOT be tracked (avoids noise on every poll)
_SKIP_PATHS = frozenset({
    "/health", "/favicon.ico", "/docs", "/openapi.json", "/redoc",
})

# Batch buffer — accumulate metrics and flush periodically
_BUFFER: deque = deque(maxlen=500)
_FLUSH_INTERVAL = 5.0  # seconds
_flush_task: asyncio.Task | None = None


async def _flush_metrics():
    """Periodically drain the buffer and write all pending metrics in one commit."""
    while True:
        await asyncio.sleep(_FLUSH_INTERVAL)
        if not _BUFFER:
            continue
        batch = []
        while _BUFFER:
            batch.append(_BUFFER.popleft())
        try:
            async with async_session_factory() as session:
                session.add_all(batch)
                await session.commit()
        except Exception as e:
            logger.debug(f"Metrics batch write failed ({len(batch)} items): {e}")


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global _flush_task

        # Ensure the background flusher is running
        if _flush_task is None or _flush_task.done():
            _flush_task = asyncio.create_task(_flush_metrics())

        start = time.perf_counter()
        response = await call_next(request)

        # Skip noisy endpoints
        if request.url.path in _SKIP_PATHS:
            return response

        latency_ms = (time.perf_counter() - start) * 1000

        # Queue the metric — zero extra latency on the response path
        _BUFFER.append(RequestMetric(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        ))

        return response
