"""
AEON Phase 10 — Bounded Telemetry Store.

Provides thread-safe and async-safe in-memory storage for request telemetry records,
with strict bounds on record count and maximum age to prevent memory bloat.
"""

import time
import asyncio
from collections import deque
from typing import Dict, Any, List, Optional
from aeon.server.telemetry.privacy import sanitize_telemetry_event


class TelemetryStore:
    """Thread-safe bounded in-memory storage for AEON operational telemetry."""

    def __init__(self, max_records: int = 10000, max_age_seconds: int = 86400, allow_content_logging: bool = False):
        self.max_records = max_records
        self.max_age_seconds = max_age_seconds
        self.allow_content_logging = allow_content_logging
        self._records: deque = deque(maxlen=max_records)
        self._lock = asyncio.Lock()

    async def record_event(self, event: Dict[str, Any]) -> None:
        """Sanitize and persist a telemetry record."""
        sanitized = sanitize_telemetry_event(event, allow_content=self.allow_content_logging)
        if "timestamp" not in sanitized:
            sanitized["timestamp"] = time.time()

        async with self._lock:
            self._records.append(sanitized)
            self._prune_expired_locked()

    async def get_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent telemetry records (newest first)."""
        async with self._lock:
            self._prune_expired_locked()
            records_list = list(self._records)
            return records_list[-limit:][::-1]

    async def get_all_records(self) -> List[Dict[str, Any]]:
        """Return all active telemetry records."""
        async with self._lock:
            self._prune_expired_locked()
            return list(self._records)

    async def clear(self) -> None:
        """Flush all records."""
        async with self._lock:
            self._records.clear()

    def _prune_expired_locked(self) -> None:
        """Remove records older than max_age_seconds."""
        now = time.time()
        cutoff = now - self.max_age_seconds
        while self._records and self._records[0].get("timestamp", now) < cutoff:
            self._records.popleft()
