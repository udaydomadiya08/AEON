"""
L1 KV / Attention-State Cache for AEON.

Provides attention KV-cache state reuse across prompt segments, avoiding redundant
prefill computation on shared tokens.
Distinguishes SIMULATED_KV_REUSE from REAL_KV_REUSE.
Conforms to AEON Specification Section 5.
"""

import time
import hashlib
from typing import Dict, List, Optional, Any
from aeon.memory.types import (
    MemoryLevel,
    MemoryRequest,
    MemoryLookupResult,
    MemoryEntry,
    ReuseDecision,
    EvictionPolicy,
    MemoryStats,
)


class KVCacheMetadata:
    """Detailed hardware and tensor descriptors for L1 KV cache."""
    def __init__(
        self,
        model_id: str,
        model_version: str,
        token_count: int,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: str = "float16",
        device: str = "mps",
        is_simulated: bool = True,
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.token_count = token_count
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.is_simulated = is_simulated
        # KV size = 2 * layers * heads * seq_len * head_dim * bytes_per_elem
        bytes_per_elem = 2 if dtype in ["float16", "bfloat16"] else (1 if dtype == "int8" else 4)
        self.size_bytes = 2 * num_layers * num_heads * token_count * head_dim * bytes_per_elem


class L1KVCache:
    """L1 KV / Attention State Cache Store."""

    def __init__(self, max_bytes: int = 20 * 1024 * 1024, is_simulated: bool = True):
        self.max_bytes = max_bytes
        self.is_simulated = is_simulated
        self._entries: Dict[str, MemoryEntry] = {}
        self._kv_descriptors: Dict[str, KVCacheMetadata] = {}
        self.stats = MemoryStats()

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def lookup(self, request: MemoryRequest) -> MemoryLookupResult:
        self.stats.total_lookups += 1
        start_time = time.perf_counter()
        
        key = self._compute_kv_key(request.query, request.model_id, request.model_version)
        entry = self._entries.get(key)
        if not entry and request.query in self._entries:
            cand = self._entries[request.query]
            if cand.model_version != request.model_version:
                self.stats.l1_misses += 1
                return MemoryLookupResult(
                    hit=False,
                    level=MemoryLevel.L1_KV,
                    lookup_cost_usd=5e-8,
                    lookup_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    explanation=f"L1 KV hit rejected: Model/tokenizer version mismatch ({cand.model_version} != {request.model_version})",
                )
            if cand.model_id is None or request.model_id is None or cand.model_id == request.model_id:
                entry = cand
                key = request.query

        lookup_latency = (time.perf_counter() - start_time) * 1000.0

        if not entry:
            self.stats.l1_misses += 1
            self.stats.total_misses += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L1_KV,
                lookup_cost_usd=5e-8,
                lookup_latency_ms=lookup_latency,
                explanation="L1 KV miss: No cached attention tensor found for prompt context",
            )

        # Check TTL / Freshness
        if entry.is_expired():
            self.invalidate(key)
            self.stats.l1_misses += 1
            self.stats.stale_hits_rejected += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L1_KV,
                lookup_cost_usd=5e-8,
                lookup_latency_ms=lookup_latency,
                explanation="L1 KV hit rejected: Cache entry expired (TTL violated)",
            )

        # Model and tokenizer compatibility check (Invariant 3)
        if entry.model_id != request.model_id or entry.model_version != request.model_version:
            self.stats.l1_misses += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L1_KV,
                explanation=f"L1 KV hit rejected: Model version mismatch ({entry.model_version} != {request.model_version})",
            )

        # Scope isolation check (Invariant 5)
        if entry.scope != request.scope or (entry.scope_id and entry.scope_id != request.scope_id):
            self.stats.l1_misses += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L1_KV,
                explanation="L1 KV hit rejected: Scope isolation mismatch",
            )

        # Update access telemetry
        entry.access_count += 1
        entry.last_accessed_at = time.time()
        self.stats.l1_hits += 1
        self.stats.total_hits += 1

        kv_desc = self._kv_descriptors.get(key)
        tokens_reused = kv_desc.token_count if kv_desc else max(1, int(len(request.query) / 4.0))
        
        # Exact FLOPs saved on prefill attention: ~ 2 * param_count * tokens_reused
        # For a 7B model, 1 token = 1.4e10 FLOPs
        compute_saved = entry.expected_compute_savings_flops
        cost_saved = entry.expected_cost_savings_usd
        latency_saved = entry.payload.get("latency_ms", 120.0) if isinstance(entry.payload, dict) else 120.0

        self.stats.total_compute_saved_flops += compute_saved
        self.stats.total_cost_saved_usd += cost_saved
        self.stats.total_tokens_saved += tokens_reused
        self.stats.total_latency_saved_ms += latency_saved

        reuse_mode = "SIMULATED_KV_REUSE" if self.is_simulated else "REAL_KV_REUSE"

        return MemoryLookupResult(
            hit=True,
            level=MemoryLevel.L1_KV,
            entry=entry,
            matched_prefix_tokens=tokens_reused,
            total_tokens=tokens_reused,
            reuse_ratio=1.0,
            semantic_similarity=1.0,
            reuse_decision=ReuseDecision.SAFE_REUSE,
            estimated_compute_saved=compute_saved,
            estimated_latency_saved_ms=latency_saved,
            estimated_cost_saved_usd=cost_saved,
            lookup_cost_usd=5e-8,
            lookup_latency_ms=lookup_latency,
            explanation=f"L1 KV exact tensor match [{reuse_mode}]: Reused attention state for {tokens_reused} tokens",
        )

    def insert(self, entry: MemoryEntry, kv_desc: Optional[KVCacheMetadata] = None) -> bool:
        """Insert attention state entry into L1 KV cache."""
        if entry.memory_level != MemoryLevel.L1_KV:
            entry.memory_level = MemoryLevel.L1_KV

        entry_size = entry.size_bytes
        # Enforce budget
        while self.total_bytes + entry_size > self.max_bytes and self._entries:
            evicted = self.evict(EvictionPolicy.LRU, target_bytes_freed=entry_size)
            if not evicted:
                break

        if self.total_bytes + entry_size > self.max_bytes:
            self.stats.admissions_rejected_count += 1
            return False

        h_key = self._compute_kv_key(entry.key, entry.model_id, entry.model_version)
        self._entries[h_key] = entry
        self._entries[entry.key] = entry
        if kv_desc:
            self._kv_descriptors[h_key] = kv_desc
            self._kv_descriptors[entry.key] = kv_desc
        self.stats.admissions_count += 1
        self.stats.total_bytes_used = self.total_bytes
        return True

    def invalidate(self, key: str, namespace: Optional[str] = None) -> int:
        count = 0
        keys_to_remove = [k for k in self._entries if k == key or (namespace and self._entries[k].namespace == namespace)]
        for k in keys_to_remove:
            del self._entries[k]
            if k in self._kv_descriptors:
                del self._kv_descriptors[k]
            count += 1
        self.stats.total_bytes_used = self.total_bytes
        return count

    def evict(self, policy: EvictionPolicy, target_bytes_freed: int = 0) -> List[str]:
        if not self._entries:
            return []

        # Sort candidate keys
        if policy == EvictionPolicy.LRU:
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].last_accessed_at)
        elif policy == EvictionPolicy.LFU:
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].access_count)
        elif policy == EvictionPolicy.VALUE_AWARE:
            # Value-aware: lowest savings per byte evicted first
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].calculate_value_density())
        else:
            sorted_keys = sorted(self._entries.keys(), key=lambda k: self._entries[k].last_accessed_at)

        freed_bytes = 0
        evicted_keys = []
        for k in sorted_keys:
            entry = self._entries[k]
            freed_bytes += entry.size_bytes
            evicted_keys.append(k)
            del self._entries[k]
            if k in self._kv_descriptors:
                del self._kv_descriptors[k]
            self.stats.evictions_count += 1
            if target_bytes_freed > 0 and freed_bytes >= target_bytes_freed:
                break

        self.stats.total_bytes_used = self.total_bytes
        return evicted_keys

    def clear(self) -> None:
        self._entries.clear()
        self._kv_descriptors.clear()
        self.stats.total_bytes_used = 0

    def get_stats(self) -> MemoryStats:
        self.stats.total_bytes_used = self.total_bytes
        total_attempts = self.stats.total_lookups
        self.stats.hit_rate = (self.stats.total_hits / total_attempts) if total_attempts > 0 else 0.0
        return self.stats

    def _compute_kv_key(self, text: str, model_id: Optional[str], model_version: str) -> str:
        h = hashlib.sha256(f"{text}|{model_id or 'default'}|{model_version}".encode()).hexdigest()
        return f"l1_kv_{h[:16]}"
