"""
AEON Memory Hierarchy Types and Protocols.

Defines schemas for multi-level memory (L1 KV, L2 Prefix, L3 Intermediate, L4 Semantic),
admission control, value-aware eviction, freshness, and scoping.
Conforms to AEON Specification Section 4, 10, 15, 18, 19, and 22.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union
from pydantic import BaseModel, Field


class MemoryLevel(str, Enum):
    L0_ACTIVE = "L0_ACTIVE"
    L1_KV = "L1_KV"
    L2_PREFIX = "L2_PREFIX"
    L3_INTERMEDIATE = "L3_INTERMEDIATE"
    L4_SEMANTIC = "L4_SEMANTIC"


class EvictionPolicy(str, Enum):
    LRU = "LRU"
    LFU = "LFU"
    TTL = "TTL"
    SIZE_AWARE = "SIZE_AWARE"
    VALUE_AWARE = "VALUE_AWARE"


class FreshnessPolicy(str, Enum):
    STRICT = "STRICT"        # Maximum 60s, requires identical context
    FRESH = "FRESH"          # Up to 1 hour, standard current facts
    EVENTUAL = "EVENTUAL"    # Up to 24 hours, general knowledge
    BEST_EFFORT = "BEST_EFFORT"  # Infinite / long TTL, static math/logic


class MemoryScope(str, Enum):
    GLOBAL = "GLOBAL"
    PROJECT = "PROJECT"
    SESSION = "SESSION"
    REQUEST = "REQUEST"
    USER = "USER"
    MODEL = "MODEL"


class ReuseDecision(str, Enum):
    SAFE_REUSE = "SAFE_REUSE"
    REUSE_WITH_VERIFICATION = "REUSE_WITH_VERIFICATION"
    REJECT_REUSE = "REJECT_REUSE"


class MemoryEntry(BaseModel):
    """Structured representation of a cached unit in the AEON memory hierarchy."""
    entry_id: str = Field(default_factory=lambda: f"mem-{uuid.uuid4().hex[:8]}")
    memory_level: MemoryLevel
    key: str
    namespace: str = "default"
    payload: Any
    payload_type: str = "text"
    created_at: float = Field(default_factory=time.time)
    last_accessed_at: float = Field(default_factory=time.time)
    access_count: int = 1
    ttl_seconds: Optional[float] = 3600.0  # Default 1 hour TTL
    size_bytes: int = 128
    model_id: Optional[str] = None
    model_version: str = "1.0.0"
    tokenizer_version: str = "1.0.0"
    prompt_hash: Optional[str] = None
    prefix_hash: Optional[str] = None
    semantic_embedding: Optional[List[float]] = None
    quality_estimate: float = 0.90
    confidence: float = 0.90
    source: str = "model_generation"
    dependencies: List[str] = Field(default_factory=list)
    freshness_policy: FreshnessPolicy = FreshnessPolicy.FRESH
    scope: MemoryScope = MemoryScope.GLOBAL
    scope_id: Optional[str] = None
    expected_compute_savings_flops: float = 1e9
    expected_cost_savings_usd: float = 0.0001
    value_density: float = 1.0  # Savings per byte ratio

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if entry exceeds TTL."""
        if self.ttl_seconds is None:
            return False
        now = current_time or time.time()
        return (now - self.created_at) > self.ttl_seconds

    def age_seconds(self, current_time: Optional[float] = None) -> float:
        now = current_time or time.time()
        return max(0.0, now - self.created_at)

    def calculate_value_density(self) -> float:
        """Expected compute/cost savings per byte of memory consumed."""
        safe_bytes = max(1, self.size_bytes)
        self.value_density = (self.expected_cost_savings_usd * 1e6) / safe_bytes
        return self.value_density


class MemoryRequest(BaseModel):
    """Query payload dispatched into the memory controller."""
    query: str
    context: Optional[str] = None
    model_id: Optional[str] = None
    model_version: str = "1.0.0"
    tokenizer_version: str = "1.0.0"
    memory_levels: List[MemoryLevel] = Field(
        default_factory=lambda: [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC]
    )
    scope: MemoryScope = MemoryScope.GLOBAL
    scope_id: Optional[str] = None
    freshness_policy: FreshnessPolicy = FreshnessPolicy.FRESH
    task_type: Optional[str] = None
    namespace: str = "default"
    max_lookup_latency_ms: float = 50.0


class MemoryLookupResult(BaseModel):
    """Result returned from hierarchical memory lookup."""
    hit: bool = False
    level: Optional[MemoryLevel] = None
    entry: Optional[MemoryEntry] = None
    matched_prefix_tokens: int = 0
    total_tokens: int = 0
    reuse_ratio: float = 0.0
    semantic_similarity: float = 0.0
    reuse_decision: ReuseDecision = ReuseDecision.REJECT_REUSE
    estimated_compute_saved: float = 0.0  # FLOPs
    estimated_latency_saved_ms: float = 0.0
    estimated_cost_saved_usd: float = 0.0
    lookup_cost_usd: float = 1e-7
    lookup_latency_ms: float = 0.5
    explanation: str = "Cache miss across all levels"


class MemoryStats(BaseModel):
    """Operational and efficiency statistics for memory hierarchy."""
    total_lookups: int = 0
    total_hits: int = 0
    total_misses: int = 0
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    l4_hits: int = 0
    l4_misses: int = 0
    hit_rate: float = 0.0
    total_bytes_used: int = 0
    total_compute_saved_flops: float = 0.0
    total_cost_saved_usd: float = 0.0
    total_tokens_saved: int = 0
    total_latency_saved_ms: float = 0.0
    evictions_count: int = 0
    admissions_count: int = 0
    admissions_rejected_count: int = 0
    false_positives_prevented: int = 0
    stale_hits_rejected: int = 0


class MemoryBudget(BaseModel):
    """Memory capacity and latency constraints."""
    max_total_bytes: int = 50 * 1024 * 1024  # 50 MB default
    max_l1_bytes: int = 15 * 1024 * 1024     # 15 MB
    max_l2_bytes: int = 15 * 1024 * 1024     # 15 MB
    max_l3_bytes: int = 10 * 1024 * 1024     # 10 MB
    max_l4_bytes: int = 10 * 1024 * 1024     # 10 MB
    max_entries: int = 10000
    eviction_policy: EvictionPolicy = EvictionPolicy.VALUE_AWARE
