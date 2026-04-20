"""Request metrics middleware — logs endpoint, method, status, latency to DB."""

import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from database import async_session_factory
from src.models import RequestMetric

logger = logging.getLogger("mindscope")

# Paths that should NOT be tracked (avoids redundant DB writes on every poll)
_SKIP_PATHS = frozenset({"/health", "/favicon.ico", "/docs", "/openapi.json", "/redoc"})


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)

        # Skip noisy endpoints to reduce unnecessary DB writes
        if request.url.path in _SKIP_PATHS:
            return response

        latency_ms = (time.perf_counter() - start) * 1000

        # Fire-and-forget DB write (don't block the response)
        try:
            async with async_session_factory() as session:
                session.add(RequestMetric(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    latency_ms=round(latency_ms, 2),
                ))
                await session.commit()
        except Exception as e:
            logger.debug(f"Metrics write failed: {e}")

        return response
