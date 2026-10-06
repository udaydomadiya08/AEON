"""
AEON Phase 10 — Server Middleware.

Provides request ID tracking, security response headers, payload size enforcement,
in-memory rate limiting, and concurrency backpressure management.
"""

import time
import uuid
import re
import asyncio
from typing import Dict, List, Optional
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from aeon.server.config import ServerConfig
from aeon.server.errors import (
    RateLimitExceededError,
    ServiceUnavailableError,
    PayloadTooLargeError
)


VALID_REQUEST_ID_REGEX = re.compile(r"^[A-Za-z0-9_\-]{4,64}$")


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Sanitizes incoming X-Request-ID or assigns a unique trace identifier."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_req_id = request.headers.get("X-Request-ID")
        if client_req_id and VALID_REQUEST_ID_REGEX.match(client_req_id):
            request_id = client_req_id
        else:
            request_id = f"req-{uuid.uuid4().hex[:16]}"

        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects standard defensive HTTP headers."""

    def __init__(self, app, config: ServerConfig):
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        if self.config.security_headers_enabled:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-AEON-Server"] = "local-orchestrator"
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Guards against denial-of-service via unbounded payload upload."""

    def __init__(self, app, config: ServerConfig):
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length_int = int(content_length)
                if length_int > self.config.max_request_bytes:
                    err = PayloadTooLargeError(
                        f"Request payload of {length_int} bytes exceeds maximum allowed limit ({self.config.max_request_bytes} bytes)."
                    )
                    return err.to_response()
            except ValueError:
                pass
        return await call_next(request)


class InMemoryRateLimiter:
    """Sliding-window rate limiter per client IP."""

    def __init__(self, max_per_minute: int = 120):
        self.max_per_minute = max_per_minute
        self.history: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, client_ip: str) -> None:
        now = time.time()
        cutoff = now - 60.0
        async with self.lock:
            timestamps = self.history.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if t > cutoff]
            if len(valid_timestamps) >= self.max_per_minute:
                raise RateLimitExceededError(
                    f"Rate limit of {self.max_per_minute} requests/minute exceeded for client."
                )
            valid_timestamps.append(now)
            self.history[client_ip] = valid_timestamps

    def reset(self) -> None:
        self.history.clear()


class ConcurrencyBackpressureManager:
    """Bounds concurrent in-flight requests and rejects excess backlog with 503."""

    def __init__(self, max_concurrency: int = 16, queue_size: int = 64, queue_timeout: float = 10.0):
        self.max_concurrency = max_concurrency
        self.queue_size = queue_size
        self.queue_timeout = queue_timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.waiting_count = 0
        self.active_count = 0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            if self.active_count >= self.max_concurrency:
                if self.waiting_count >= self.queue_size:
                    raise ServiceUnavailableError(
                        "Server overload: backpressure queue capacity exceeded. Please retry later."
                    )
                self.waiting_count += 1
            else:
                self.active_count += 1
                return

        try:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=self.queue_timeout)
            async with self.lock:
                self.waiting_count -= 1
                self.active_count += 1
        except asyncio.TimeoutError:
            async with self.lock:
                self.waiting_count -= 1
            raise ServiceUnavailableError(
                "Request queued waiting for concurrency slot timed out."
            )
        except Exception:
            async with self.lock:
                self.waiting_count -= 1
            raise

    async def release(self) -> None:
        async with self.lock:
            self.active_count = max(0, self.active_count - 1)
        self.semaphore.release()
