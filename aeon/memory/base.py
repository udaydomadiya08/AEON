"""
Base Memory Store Protocol and Abstract Interfaces for AEON.
Conforms to AEON Specification Section 3.
"""

from typing import Protocol, List, Optional, runtime_checkable
from aeon.memory.types import (
    MemoryRequest,
    MemoryLookupResult,
    MemoryEntry,
    EvictionPolicy,
    MemoryStats,
)


@runtime_checkable
class MemoryStore(Protocol):
    """Abstract protocol for individual memory tiers and integrated stores."""

    def lookup(self, request: MemoryRequest) -> MemoryLookupResult:
        """Query the memory store for matching cached states or results."""
        ...

    def insert(self, entry: MemoryEntry) -> bool:
        """Insert a new memory entry into the store. Returns True if admitted."""
        ...

    def invalidate(self, key: str, namespace: Optional[str] = None) -> int:
        """Invalidate entries by key or prefix. Returns count of invalidated items."""
        ...

    def evict(self, policy: EvictionPolicy, target_bytes_freed: int = 0) -> List[str]:
        """Evict entries according to configured policy to free memory."""
        ...

    def clear(self) -> None:
        """Clear all entries from the store."""
        ...

    def get_stats(self) -> MemoryStats:
        """Retrieve telemetry and operational statistics."""
        ...
