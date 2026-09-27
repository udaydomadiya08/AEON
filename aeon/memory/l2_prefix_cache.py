"""
L2 Prefix / Prompt Cache for AEON.

Provides longest common prefix matching and reuse across shared system prompts
and multi-turn contexts.
Conforms to AEON Specification Section 6.
"""

import time
import hashlib
from typing import Dict, List, Optional, Tuple
from aeon.memory.types import (
    MemoryLevel,
    MemoryRequest,
    MemoryLookupResult,
    MemoryEntry,
    ReuseDecision,
    EvictionPolicy,
    MemoryStats,
)


class L2PrefixCache:
    """L2 Prefix and Prompt State Cache supporting Longest Common Prefix (LCP) matching."""

    def __init__(self, max_bytes: int = 20 * 1024 * 1024, min_prefix_tokens: int = 8):
        self.max_bytes = max_bytes
        self.min_prefix_tokens = min_prefix_tokens
        # Maps prefix_hash -> MemoryEntry
        self._entries: Dict[str, MemoryEntry] = {}
        # List of stored (prefix_text, prefix_hash, token_count)
        self._prefix_index: List[Tuple[str, str, int]] = []
        self.stats = MemoryStats()

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def lookup(self, request: MemoryRequest) -> MemoryLookupResult:
        self.stats.total_lookups += 1
        start_time = time.perf_counter()

        query_text = request.query.strip()
        total_tokens = max(1, int(len(query_text) / 4.0))

        # Find longest matching prefix among cached entries
        best_match: Optional[MemoryEntry] = None
        longest_reused_tokens = 0
        longest_reused_chars = 0

        for prefix_text, prefix_hash, token_count in self._prefix_index:
            if query_text.startswith(prefix_text) and token_count >= self.min_prefix_tokens:
                entry = self._entries.get(prefix_hash)
                if not entry:
                    continue

                # Check expiration
                if entry.is_expired():
                    continue

                # Model and tokenizer compatibility (Invariant 3)
                if entry.model_id and request.model_id and entry.model_id != request.model_id:
                    continue
                if entry.tokenizer_version != request.tokenizer_version:
                    continue

                # Scope check (Invariant 5)
                if entry.scope != request.scope or (entry.scope_id and entry.scope_id != request.scope_id):
                    continue

                if token_count > longest_reused_tokens:
                    longest_reused_tokens = token_count
                    longest_reused_chars = len(prefix_text)
                    best_match = entry

        lookup_latency = (time.perf_counter() - start_time) * 1000.0

        if not best_match or longest_reused_tokens < self.min_prefix_tokens:
            self.stats.l2_misses += 1
            self.stats.total_misses += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L2_PREFIX,
                total_tokens=total_tokens,
                lookup_cost_usd=8e-8,
                lookup_latency_ms=lookup_latency,
                explanation="L2 Prefix miss: No reusable prefix exceeding minimum token threshold",
            )

        # Update access telemetry
        best_match.access_count += 1
        best_match.last_accessed_at = time.time()
        self.stats.l2_hits += 1
        self.stats.total_hits += 1

        reuse_ratio = min(1.0, longest_reused_tokens / float(total_tokens))
        
        # Calculate quantitative savings based on fraction of prefix prefill avoided
        # Standard ~ 2 * param_count * reused_tokens
        compute_saved = 2.0 * (1.5e9) * longest_reused_tokens  # Base 1.5B estimate
        cost_saved = (longest_reused_tokens / 1_000_000.0) * 0.15  # ~$0.15/1M input tokens
        latency_saved = max(5.0, (longest_reused_tokens / 220.0) * 1000.0)  # Prefill time saved

        self.stats.total_compute_saved_flops += compute_saved
        self.stats.total_cost_saved_usd += cost_saved
        self.stats.total_tokens_saved += longest_reused_tokens
        self.stats.total_latency_saved_ms += latency_saved

        return MemoryLookupResult(
            hit=True,
            level=MemoryLevel.L2_PREFIX,
            entry=best_match,
            matched_prefix_tokens=longest_reused_tokens,
            total_tokens=total_tokens,
            reuse_ratio=round(reuse_ratio, 4),
            semantic_similarity=round(reuse_ratio, 4),
            reuse_decision=ReuseDecision.SAFE_REUSE,
            estimated_compute_saved=compute_saved,
            estimated_latency_saved_ms=round(latency_saved, 2),
            estimated_cost_saved_usd=round(cost_saved, 7),
            lookup_cost_usd=8e-8,
            lookup_latency_ms=lookup_latency,
            explanation=f"L2 Prefix match: Reused {longest_reused_tokens}/{total_tokens} tokens ({reuse_ratio*100:.1f}%)",
        )

    def insert(self, entry: MemoryEntry) -> bool:
        """Insert a reusable prefix entry."""
        if entry.memory_level != MemoryLevel.L2_PREFIX:
            entry.memory_level = MemoryLevel.L2_PREFIX

        prefix_text = str(entry.payload) if isinstance(entry.payload, str) else entry.key
        token_count = max(1, int(len(prefix_text) / 4.0))
        if token_count < self.min_prefix_tokens:
            return False

        # Budget enforcement
        while self.total_bytes + entry.size_bytes > self.max_bytes and self._entries:
            self.evict(EvictionPolicy.LRU, target_bytes_freed=entry.size_bytes)

        if self.total_bytes + entry.size_bytes > self.max_bytes:
            self.stats.admissions_rejected_count += 1
            return False

        h = hashlib.sha256(prefix_text.encode()).hexdigest()[:16]
        entry.prefix_hash = h
        if not entry.key:
            entry.key = h

        self._entries[h] = entry
        self._entries[entry.key] = entry
        # Update prefix index sorted by length descending for greedy longest match
        self._prefix_index = [p for p in self._prefix_index if p[1] != h]
        self._prefix_index.append((prefix_text, h, token_count))
        self._prefix_index.sort(key=lambda x: x[2], reverse=True)

        self.stats.admissions_count += 1
        self.stats.total_bytes_used = self.total_bytes
        return True

    def invalidate(self, key: str, namespace: Optional[str] = None) -> int:
        count = 0
        to_del = [k for k in self._entries if k == key or (namespace and self._entries[k].namespace == namespace)]
        for k in to_del:
            del self._entries[k]
            count += 1
        self._prefix_index = [p for p in self._prefix_index if p[1] not in to_del]
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

        self._prefix_index = [p for p in self._prefix_index if p[1] not in evicted]
        self.stats.total_bytes_used = self.total_bytes
        return evicted

    def clear(self) -> None:
        self._entries.clear()
        self._prefix_index.clear()
        self.stats.total_bytes_used = 0

    def get_stats(self) -> MemoryStats:
        self.stats.total_bytes_used = self.total_bytes
        total_attempts = self.stats.total_lookups
        self.stats.hit_rate = (self.stats.total_hits / total_attempts) if total_attempts > 0 else 0.0
        return self.stats
