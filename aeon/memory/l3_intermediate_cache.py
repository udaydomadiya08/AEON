"""
L3 Intermediate Computation Cache for AEON.

Caches deterministic and intermediate artifacts (tool outputs, mathematical subexpressions,
tokenizations, document chunks, classification results) organized by typed namespaces.
Supports dependency-driven invalidation.
Conforms to AEON Specification Section 7.
"""

import time
import hashlib
from typing import Dict, List, Optional, Set, Any
from aeon.memory.types import (
    MemoryLevel,
    MemoryRequest,
    MemoryLookupResult,
    MemoryEntry,
    ReuseDecision,
    EvictionPolicy,
    MemoryStats,
)


class L3IntermediateCache:
    """L3 Intermediate Computation Cache with typed namespaces and dependency tracking."""

    VALID_NAMESPACES = {
        "tokenization",
        "embedding",
        "retrieval",
        "tool",
        "classification",
        "difficulty",
        "intermediate",
        "math_subexpression",
        "default",
    }

    def __init__(self, max_bytes: int = 15 * 1024 * 1024):
        self.max_bytes = max_bytes
        # Map (namespace, key) -> MemoryEntry
        self._entries: Dict[str, MemoryEntry] = {}
        # Dependency map: source_id -> Set of entry keys
        self._dependency_map: Dict[str, Set[str]] = {}
        self.stats = MemoryStats()

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def lookup(self, request: MemoryRequest) -> MemoryLookupResult:
        self.stats.total_lookups += 1
        start_time = time.perf_counter()

        namespace = request.namespace if request.namespace in self.VALID_NAMESPACES else "default"
        key = self._format_key(namespace, request.query)

        entry = self._entries.get(key)
        lookup_latency = (time.perf_counter() - start_time) * 1000.0

        if not entry:
            self.stats.l3_misses += 1
            self.stats.total_misses += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L3_INTERMEDIATE,
                lookup_cost_usd=2e-8,
                lookup_latency_ms=lookup_latency,
                explanation=f"L3 Intermediate miss: No entry in namespace '{namespace}' for key",
            )

        # TTL check
        if entry.is_expired():
            self.invalidate(entry.key, namespace=entry.namespace)
            self.stats.l3_misses += 1
            self.stats.stale_hits_rejected += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L3_INTERMEDIATE,
                lookup_cost_usd=2e-8,
                lookup_latency_ms=lookup_latency,
                explanation="L3 Intermediate hit rejected: Expired TTL",
            )

        # Scope isolation check
        if entry.scope != request.scope or (entry.scope_id and entry.scope_id != request.scope_id):
            self.stats.l3_misses += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L3_INTERMEDIATE,
                explanation="L3 Intermediate hit rejected: Scope isolation mismatch",
            )

        # Update telemetry
        entry.access_count += 1
        entry.last_accessed_at = time.time()
        self.stats.l3_hits += 1
        self.stats.total_hits += 1

        compute_saved = entry.expected_compute_savings_flops
        cost_saved = entry.expected_cost_savings_usd
        latency_saved = 15.0

        self.stats.total_compute_saved_flops += compute_saved
        self.stats.total_cost_saved_usd += cost_saved
        self.stats.total_latency_saved_ms += latency_saved

        return MemoryLookupResult(
            hit=True,
            level=MemoryLevel.L3_INTERMEDIATE,
            entry=entry,
            matched_prefix_tokens=0,
            total_tokens=max(1, int(len(request.query) / 4.0)),
            reuse_ratio=1.0,
            semantic_similarity=1.0,
            reuse_decision=ReuseDecision.SAFE_REUSE,
            estimated_compute_saved=compute_saved,
            estimated_latency_saved_ms=latency_saved,
            estimated_cost_saved_usd=cost_saved,
            lookup_cost_usd=2e-8,
            lookup_latency_ms=lookup_latency,
            explanation=f"L3 Intermediate hit in namespace '{entry.namespace}': Deterministic subcomputation reused",
        )

    def insert(self, entry: MemoryEntry) -> bool:
        """Insert intermediate artifact into L3 cache."""
        if entry.memory_level != MemoryLevel.L3_INTERMEDIATE:
            entry.memory_level = MemoryLevel.L3_INTERMEDIATE

        if entry.namespace not in self.VALID_NAMESPACES:
            entry.namespace = "default"

        key = self._format_key(entry.namespace, entry.key)
        entry.key = key

        # Budget check
        while self.total_bytes + entry.size_bytes > self.max_bytes and self._entries:
            self.evict(EvictionPolicy.VALUE_AWARE, target_bytes_freed=entry.size_bytes)

        if self.total_bytes + entry.size_bytes > self.max_bytes:
            self.stats.admissions_rejected_count += 1
            return False

        self._entries[key] = entry

        # Track dependencies
        for dep in entry.dependencies:
            if dep not in self._dependency_map:
                self._dependency_map[dep] = set()
            self._dependency_map[dep].add(key)

        self.stats.admissions_count += 1
        self.stats.total_bytes_used = self.total_bytes
        return True

    def invalidate_by_dependency(self, source_dependency: str) -> int:
        """Cascade invalidation when source data changes (e.g., document update)."""
        dependent_keys = self._dependency_map.pop(source_dependency, set())
        count = 0
        for k in dependent_keys:
            if k in self._entries:
                del self._entries[k]
                count += 1
        self.stats.total_bytes_used = self.total_bytes
        return count

    def invalidate(self, key: str, namespace: Optional[str] = None) -> int:
        count = 0
        formatted_key = self._format_key(namespace or "default", key)
        if formatted_key in self._entries:
            del self._entries[formatted_key]
            count += 1
        else:
            # Check direct key match
            for k in list(self._entries.keys()):
                if k == key or k.endswith(f"/{key}"):
                    del self._entries[k]
                    count += 1
        self.stats.total_bytes_used = self.total_bytes
        return count

    def evict(self, policy: EvictionPolicy, target_bytes_freed: int = 0) -> List[str]:
        if not self._entries:
            return []

        if policy == EvictionPolicy.LRU:
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].last_accessed_at)
        elif policy == EvictionPolicy.LFU:
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].access_count)
        elif policy == EvictionPolicy.VALUE_AWARE:
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].calculate_value_density())
        else:
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].last_accessed_at)

        freed = 0
        evicted = []
        for k in sorted_keys:
            freed += self._entries[k].size_bytes
            evicted.append(k)
            del self._entries[k]
            self.stats.evictions_count += 1
            if target_bytes_freed > 0 and freed >= target_bytes_freed:
                break

        self.stats.total_bytes_used = self.total_bytes
        return evicted

    def clear(self) -> None:
        self._entries.clear()
        self._dependency_map.clear()
        self.stats.total_bytes_used = 0

    def get_stats(self) -> MemoryStats:
        self.stats.total_bytes_used = self.total_bytes
        total_attempts = self.stats.total_lookups
        self.stats.hit_rate = (self.stats.total_hits / total_attempts) if total_attempts > 0 else 0.0
        return self.stats

    def _format_key(self, namespace: str, raw_key: str) -> str:
        h = hashlib.sha256(raw_key.strip().encode()).hexdigest()[:16]
        return f"{namespace}/{h}"
