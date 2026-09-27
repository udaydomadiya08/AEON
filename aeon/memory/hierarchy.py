"""
Unified AEON Multi-Level Memory Hierarchy.

Orchestrates L1 KV, L2 Prefix, L3 Intermediate, and L4 Semantic tiers with
economic admission control, value-aware eviction, freshness enforcement,
and telemetry accounting.
Conforms to AEON Specification Section 13, 16, 17, 18, 19, 21, and 39.
"""

import time
import os
import json
from typing import Dict, List, Optional, Any
from aeon.memory.types import (
    MemoryLevel,
    MemoryRequest,
    MemoryLookupResult,
    MemoryEntry,
    MemoryStats,
    MemoryBudget,
    ReuseDecision,
    EvictionPolicy,
    FreshnessPolicy,
    MemoryScope,
)
from aeon.memory.l1_kv_cache import L1KVCache
from aeon.memory.l2_prefix_cache import L2PrefixCache
from aeon.memory.l3_intermediate_cache import L3IntermediateCache
from aeon.memory.l4_semantic_cache import L4SemanticCache
from aeon.memory.admission import CacheAdmissionController
from aeon.memory.economics import MemoryEconomicsEstimator


class AEONMemoryHierarchy:
    """
    Unified multi-tier memory system for AEON.
    Evaluates: L1 KV -> L2 Prefix -> L3 Intermediate -> L4 Semantic.
    """

    def __init__(
        self,
        budget: Optional[MemoryBudget] = None,
        is_simulated_kv: bool = True,
        persistence_path: Optional[str] = None,
    ):
        self.budget = budget or MemoryBudget()
        self.persistence_path = persistence_path
        self.l1 = L1KVCache(max_bytes=self.budget.max_l1_bytes, is_simulated=is_simulated_kv)
        self.l2 = L2PrefixCache(max_bytes=self.budget.max_l2_bytes)
        self.l3 = L3IntermediateCache(max_bytes=self.budget.max_l3_bytes)
        self.l4 = L4SemanticCache(max_bytes=self.budget.max_l4_bytes)
        
        self.admission_controller = CacheAdmissionController()
        self.economics_estimator = MemoryEconomicsEstimator()

        if self.persistence_path and os.path.exists(self.persistence_path):
            self.load_state(self.persistence_path)

    @property
    def total_bytes_used(self) -> int:
        return self.l1.total_bytes + self.l2.total_bytes + self.l3.total_bytes + self.l4.total_bytes

    def lookup(self, request: MemoryRequest) -> MemoryLookupResult:
        """
        Hierarchical query traversal:
        1. L1 KV exact prompt tensor match
        2. L2 Longest reusable prefix match
        3. L3 Intermediate artifact match (if namespace specified)
        4. L4 Semantic result match
        """
        start_time = time.perf_counter()

        result = None
        # 1. L1 KV lookup (if enabled in request)
        if MemoryLevel.L1_KV in request.memory_levels:
            res_l1 = self.l1.lookup(request)
            if res_l1.hit:
                result = res_l1

        # 2. L2 Prefix lookup (if enabled in request)
        if result is None and MemoryLevel.L2_PREFIX in request.memory_levels:
            res_l2 = self.l2.lookup(request)
            if res_l2.hit:
                result = res_l2

        # 3. L3 Intermediate lookup (if namespace provided or applicable)
        if result is None and MemoryLevel.L3_INTERMEDIATE in request.memory_levels and request.namespace != "default":
            res_l3 = self.l3.lookup(request)
            if res_l3.hit:
                result = res_l3

        # 4. L4 Semantic lookup
        if result is None and MemoryLevel.L4_SEMANTIC in request.memory_levels:
            res_l4 = self.l4.lookup(request)
            if res_l4.hit:
                result = res_l4

        if result is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            result = MemoryLookupResult(
                hit=False,
                level=None,
                total_tokens=max(1, int(len(request.query) / 4.0)),
                lookup_cost_usd=2e-7,
                lookup_latency_ms=elapsed_ms,
                explanation="Cache miss across all memory levels (L1, L2, L3, L4)",
            )

        if self.persistence_path:
            self.dump_state()
        return result

    def admit_and_store(
        self,
        request: MemoryRequest,
        payload: Any,
        level: MemoryLevel = MemoryLevel.L4_SEMANTIC,
        quality_estimate: float = 0.90,
        confidence: float = 0.90,
        ttl_seconds: float = 3600.0,
        dependencies: Optional[List[str]] = None,
    ) -> bool:
        """
        Applies economic admission control and inserts entry into designated tier.
        """
        size_bytes = max(64, len(str(payload).encode("utf-8")))
        entry = MemoryEntry(
            memory_level=level,
            key=request.query.strip(),
            namespace=request.namespace,
            payload=payload,
            ttl_seconds=ttl_seconds,
            size_bytes=size_bytes,
            model_id=request.model_id,
            model_version=request.model_version,
            tokenizer_version=request.tokenizer_version,
            quality_estimate=quality_estimate,
            confidence=confidence,
            dependencies=dependencies or [],
            freshness_policy=request.freshness_policy,
            scope=request.scope,
            scope_id=request.scope_id,
            expected_compute_savings_flops=2.0 * 7e9 * max(15, int(size_bytes / 4.0)),
            expected_cost_savings_usd=0.00003,
        )
        entry.calculate_value_density()

        # Economic admission check (Section 10)
        admit, reason = self.admission_controller.should_admit(
            entry,
            task_type=request.task_type,
            query_text=request.query,
        )
        if not admit:
            return False

        # Route to appropriate tier
        inserted = False
        if level == MemoryLevel.L1_KV:
            inserted = self.l1.insert(entry)
        elif level == MemoryLevel.L2_PREFIX:
            inserted = self.l2.insert(entry)
        elif level == MemoryLevel.L3_INTERMEDIATE:
            inserted = self.l3.insert(entry)
        else:
            inserted = self.l4.insert(entry)

        if inserted and self.persistence_path:
            self.dump_state()
        return inserted

    def warm_prefix(self, prefix_text: str, model_id: str = "aeon-small-1.5b") -> bool:
        """Warms L2 prefix cache with a high-frequency system prompt."""
        entry = MemoryEntry(
            memory_level=MemoryLevel.L2_PREFIX,
            key=prefix_text,
            payload=prefix_text,
            size_bytes=len(prefix_text.encode("utf-8")),
            model_id=model_id,
            ttl_seconds=86400.0,  # 24h
            scope=MemoryScope.GLOBAL,
        )
        res = self.l2.insert(entry)
        if res and self.persistence_path:
            self.dump_state()
        return res

    def invalidate(self, key: str, namespace: Optional[str] = None) -> int:
        """Invalidates matching entries across all tiers."""
        c1 = self.l1.invalidate(key, namespace)
        c2 = self.l2.invalidate(key, namespace)
        c3 = self.l3.invalidate(key, namespace)
        c4 = self.l4.invalidate(key, namespace)
        tot = c1 + c2 + c3 + c4
        if tot > 0 and self.persistence_path:
            self.dump_state()
        return tot

    def clear(self) -> None:
        """Flush all memory tiers."""
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        self.l4.clear()
        if self.persistence_path and os.path.exists(self.persistence_path):
            try:
                os.remove(self.persistence_path)
            except OSError:
                pass

    def dump_state(self, path: Optional[str] = None) -> None:
        """Persists memory state to disk for cross-invocation persistence."""
        target_path = path or self.persistence_path
        if not target_path:
            return
        data = {
            "l1": [e.model_dump(mode="json") for e in self.l1._entries.values()],
            "l2": [e.model_dump(mode="json") for e in self.l2._entries.values()],
            "l3": [e.model_dump(mode="json") for e in self.l3._entries.values()],
            "l4": [e.model_dump(mode="json") for e in self.l4._entries.values()],
            "l1_stats": self.l1.get_stats().model_dump(mode="json"),
            "l2_stats": self.l2.get_stats().model_dump(mode="json"),
            "l3_stats": self.l3.get_stats().model_dump(mode="json"),
            "l4_stats": self.l4.get_stats().model_dump(mode="json"),
        }
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_state(self, path: Optional[str] = None) -> None:
        """Loads persisted memory state from disk."""
        target_path = path or self.persistence_path
        if not target_path or not os.path.exists(target_path):
            return
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("l1", []):
                self.l1.insert(MemoryEntry(**item))
            for item in data.get("l2", []):
                self.l2.insert(MemoryEntry(**item))
            for item in data.get("l3", []):
                self.l3.insert(MemoryEntry(**item))
            for item in data.get("l4", []):
                self.l4.insert(MemoryEntry(**item))
            if "l1_stats" in data:
                self.l1.stats = MemoryStats(**data["l1_stats"])
            if "l2_stats" in data:
                self.l2.stats = MemoryStats(**data["l2_stats"])
            if "l3_stats" in data:
                self.l3.stats = MemoryStats(**data["l3_stats"])
            if "l4_stats" in data:
                self.l4.stats = MemoryStats(**data["l4_stats"])
        except Exception:
            pass

    def stats(self) -> MemoryStats:
        """Alias for get_stats() conforming to Section 32."""
        return self.get_stats()

    def get_stats(self) -> MemoryStats:
        """Aggregate operational statistics across all tiers."""
        s1 = self.l1.get_stats()
        s2 = self.l2.get_stats()
        s3 = self.l3.get_stats()
        s4 = self.l4.get_stats()

        total_lookups = s1.total_lookups + s2.total_lookups + s3.total_lookups + s4.total_lookups
        total_hits = s1.total_hits + s2.total_hits + s3.total_hits + s4.total_hits
        total_misses = s1.total_misses + s2.total_misses + s3.total_misses + s4.total_misses
        hit_rate = (total_hits / total_lookups) if total_lookups > 0 else 0.0

        return MemoryStats(
            total_lookups=total_lookups,
            total_hits=total_hits,
            total_misses=total_misses,
            l1_hits=s1.l1_hits,
            l1_misses=s1.l1_misses,
            l2_hits=s2.l2_hits,
            l2_misses=s2.l2_misses,
            l3_hits=s3.l3_hits,
            l3_misses=s3.l3_misses,
            l4_hits=s4.l4_hits,
            l4_misses=s4.l4_misses,
            hit_rate=round(hit_rate, 4),
            total_bytes_used=self.total_bytes_used,
            total_compute_saved_flops=s1.total_compute_saved_flops + s2.total_compute_saved_flops + s3.total_compute_saved_flops + s4.total_compute_saved_flops,
            total_cost_saved_usd=round(s1.total_cost_saved_usd + s2.total_cost_saved_usd + s3.total_cost_saved_usd + s4.total_cost_saved_usd, 6),
            total_tokens_saved=s1.total_tokens_saved + s2.total_tokens_saved + s3.total_tokens_saved + s4.total_tokens_saved,
            total_latency_saved_ms=round(s1.total_latency_saved_ms + s2.total_latency_saved_ms + s3.total_latency_saved_ms + s4.total_latency_saved_ms, 2),
            evictions_count=s1.evictions_count + s2.evictions_count + s3.evictions_count + s4.evictions_count,
            admissions_count=s1.admissions_count + s2.admissions_count + s3.admissions_count + s4.admissions_count,
            admissions_rejected_count=s1.admissions_rejected_count + s2.admissions_rejected_count + s3.admissions_rejected_count + s4.admissions_rejected_count,
            false_positives_prevented=s4.false_positives_prevented,
            stale_hits_rejected=s1.stale_hits_rejected + s2.stale_hits_rejected + s3.stale_hits_rejected,
        )

    def inspect(self) -> Dict[str, Any]:
        """Provides an inspection summary of memory hierarchy contents."""
        return {
            "total_bytes_used": self.total_bytes_used,
            "max_budget_bytes": self.budget.max_total_bytes,
            "tiers": {
                "L1_KV": {"entries": len(self.l1._entries), "bytes": self.l1.total_bytes},
                "L2_Prefix": {"entries": len(self.l2._entries), "bytes": self.l2.total_bytes},
                "L3_Intermediate": {"entries": len(self.l3._entries), "bytes": self.l3.total_bytes},
                "L4_Semantic": {"entries": len(self.l4._entries), "bytes": self.l4.total_bytes},
            },
            "stats": self.get_stats().model_dump(),
        }


# Global default memory hierarchy instance with persistent backing
global_memory_hierarchy = AEONMemoryHierarchy(persistence_path=".aeon_memory.json")
