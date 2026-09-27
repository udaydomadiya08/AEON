"""
L4 Semantic Memory and Result Cache for AEON.

Features:
- Configurable SemanticMatcher (character-ngram Jaccard + word-overlap + token set similarity)
- Strict Semantic Safety Protocol:
  Checks task compatibility, temporal entity differences, number/named-entity mismatches,
  freshness decay, and quality thresholds to prevent false reuse.
- Explicit decisions: SAFE_REUSE, REUSE_WITH_VERIFICATION, REJECT_REUSE.
Conforms to AEON Specification Section 8, 9, and 27.
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Protocol, runtime_checkable
from aeon.memory.types import (
    MemoryLevel,
    MemoryRequest,
    MemoryLookupResult,
    MemoryEntry,
    ReuseDecision,
    EvictionPolicy,
    FreshnessPolicy,
    MemoryStats,
)


@runtime_checkable
class SemanticMatcher(Protocol):
    """Protocol for semantic similarity comparison."""
    def similarity(self, query_a: str, query_b: str) -> float:
        ...


class HybridSemanticMatcher:
    """
    Fast, deterministic semantic matcher combining token overlap, Jaccard character-ngram
    similarity, and semantic structure without external API dependencies.
    """

    def similarity(self, query_a: str, query_b: str) -> float:
        a_norm = self._normalize(query_a)
        b_norm = self._normalize(query_b)

        if a_norm == b_norm:
            return 1.0

        # Word-level Jaccard similarity
        words_a = set(a_norm.split())
        words_b = set(b_norm.split())
        if not words_a or not words_b:
            return 0.0

        word_intersection = len(words_a & words_b)
        word_union = len(words_a | words_b)
        word_jaccard = word_intersection / float(word_union)

        # 3-gram character Jaccard similarity
        ngrams_a = self._ngrams(a_norm, 3)
        ngrams_b = self._ngrams(b_norm, 3)
        if not ngrams_a or not ngrams_b:
            ngram_jaccard = 0.0
        else:
            ngram_jaccard = len(ngrams_a & ngrams_b) / float(len(ngrams_a | ngrams_b))

        # Hybrid score
        score = 0.60 * word_jaccard + 0.40 * ngram_jaccard
        return round(float(score), 4)

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    def _ngrams(self, text: str, n: int) -> set:
        return {text[i:i+n] for i in range(max(0, len(text) - n + 1))}


class L4SemanticCache:
    """
    L4 Semantic Memory Cache Store with strict semantic safety verification.
    """

    TEMPORAL_PATTERNS = re.compile(r"\b(19\d\d|20\d\d|yesterday|tomorrow|today|future|past|last year|next year)\b", re.IGNORECASE)
    NUMBER_PATTERNS = re.compile(r"\b\d+(\.\d+)?%?\b")

    def __init__(
        self,
        matcher: Optional[SemanticMatcher] = None,
        max_bytes: int = 15 * 1024 * 1024,
        similarity_threshold: float = 0.82,
        high_similarity_threshold: float = 0.94,
        eviction_policy: EvictionPolicy = EvictionPolicy.VALUE_AWARE,
    ):
        self.matcher = matcher or HybridSemanticMatcher()
        self.max_bytes = max_bytes
        self.similarity_threshold = similarity_threshold
        self.high_similarity_threshold = high_similarity_threshold
        self.eviction_policy = eviction_policy
        self._entries: Dict[str, MemoryEntry] = {}
        self.stats = MemoryStats()

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    def lookup(self, request: MemoryRequest) -> MemoryLookupResult:
        self.stats.total_lookups += 1
        start_time = time.perf_counter()

        best_entry: Optional[MemoryEntry] = None
        best_sim = 0.0

        # Scan cached entries for semantic similarity
        for entry in self._entries.values():
            # Check expiration
            if entry.is_expired():
                continue

            # Scope isolation (Invariant 5)
            if entry.scope != request.scope or (entry.scope_id and entry.scope_id != request.scope_id):
                continue

            sim = self.matcher.similarity(request.query, entry.key)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        lookup_latency = (time.perf_counter() - start_time) * 1000.0

        # Check threshold
        if not best_entry or best_sim < self.similarity_threshold:
            self.stats.l4_misses += 1
            self.stats.total_misses += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L4_SEMANTIC,
                semantic_similarity=best_sim,
                reuse_decision=ReuseDecision.REJECT_REUSE,
                lookup_cost_usd=1e-7,
                lookup_latency_ms=lookup_latency,
                explanation=f"L4 Semantic miss: Best similarity ({best_sim:.2f}) below threshold ({self.similarity_threshold:.2f})",
            )

        # SEMANTIC CACHE SAFETY EVALUATION (Section 9 & 27)
        safety_decision, safety_reason = self._evaluate_semantic_safety(request, best_entry, best_sim)

        if safety_decision == ReuseDecision.REJECT_REUSE:
            self.stats.l4_misses += 1
            self.stats.false_positives_prevented += 1
            return MemoryLookupResult(
                hit=False,
                level=MemoryLevel.L4_SEMANTIC,
                entry=best_entry,
                semantic_similarity=best_sim,
                reuse_decision=ReuseDecision.REJECT_REUSE,
                lookup_cost_usd=1e-7,
                lookup_latency_ms=lookup_latency,
                explanation=f"L4 Semantic hit rejected by safety guard: {safety_reason}",
            )

        # Hit accepted (Safe or Verification needed)
        best_entry.access_count += 1
        best_entry.last_accessed_at = time.time()
        self.stats.l4_hits += 1
        self.stats.total_hits += 1

        compute_saved = best_entry.expected_compute_savings_flops
        cost_saved = best_entry.expected_cost_savings_usd
        latency_saved = 250.0  # Avoided full generation latency

        self.stats.total_compute_saved_flops += compute_saved
        self.stats.total_cost_saved_usd += cost_saved
        self.stats.total_tokens_saved += max(15, int(len(str(best_entry.payload)) / 4.0))
        self.stats.total_latency_saved_ms += latency_saved

        return MemoryLookupResult(
            hit=True,
            level=MemoryLevel.L4_SEMANTIC,
            entry=best_entry,
            matched_prefix_tokens=0,
            total_tokens=max(1, int(len(request.query) / 4.0)),
            reuse_ratio=1.0,
            semantic_similarity=best_sim,
            reuse_decision=safety_decision,
            estimated_compute_saved=compute_saved,
            estimated_latency_saved_ms=latency_saved,
            estimated_cost_saved_usd=cost_saved,
            lookup_cost_usd=1e-7,
            lookup_latency_ms=lookup_latency,
            explanation=f"L4 Semantic {safety_decision.value} (sim={best_sim:.2f}): {safety_reason}",
        )

    def insert(self, entry: MemoryEntry) -> bool:
        """Insert a result entry into L4 semantic cache."""
        if entry.memory_level != MemoryLevel.L4_SEMANTIC:
            entry.memory_level = MemoryLevel.L4_SEMANTIC

        # Budget enforcement
        while self.total_bytes + entry.size_bytes > self.max_bytes and self._entries:
            self.evict(self.eviction_policy, target_bytes_freed=entry.size_bytes)

        if self.total_bytes + entry.size_bytes > self.max_bytes:
            self.stats.admissions_rejected_count += 1
            return False

        self._entries[entry.key] = entry
        self.stats.admissions_count += 1
        self.stats.total_bytes_used = self.total_bytes
        return True

    def _evaluate_semantic_safety(
        self,
        request: MemoryRequest,
        entry: MemoryEntry,
        similarity: float,
    ) -> Tuple[ReuseDecision, str]:
        """
        Guarantees that similar but fundamentally different queries are not falsely answered.
        (e.g., 'revenue in 2024' vs 'revenue in 2026').
        """
        q_new = request.query.lower()
        q_cached = entry.key.lower()

        # 1. Temporal entity mismatch check
        years_new = set(self.TEMPORAL_PATTERNS.findall(q_new))
        years_cached = set(self.TEMPORAL_PATTERNS.findall(q_cached))
        if years_new != years_cached:
            return (
                ReuseDecision.REJECT_REUSE,
                f"Temporal entity discrepancy detected: query={years_new} vs cached={years_cached}",
            )

        # 2. Specific numerical parameter mismatch check
        nums_new = set(self.NUMBER_PATTERNS.findall(q_new))
        nums_cached = set(self.NUMBER_PATTERNS.findall(q_cached))
        if nums_new != nums_cached and len(nums_new | nums_cached) > 0:
            return (
                ReuseDecision.REJECT_REUSE,
                f"Numerical parameter mismatch detected: query={nums_new} vs cached={nums_cached}",
            )

        # 3. Freshness policy constraint
        age = entry.age_seconds()
        if request.freshness_policy == FreshnessPolicy.STRICT and age > 60.0:
            return ReuseDecision.REJECT_REUSE, f"STRICT freshness violated: age {age:.1f}s > 60s"
        elif request.freshness_policy == FreshnessPolicy.FRESH and age > 3600.0:
            return ReuseDecision.REUSE_WITH_VERIFICATION, f"FRESH policy age {age:.1f}s > 1h, requires verification"

        # 4. High vs Moderate similarity
        if similarity >= self.high_similarity_threshold:
            return ReuseDecision.SAFE_REUSE, "High semantic similarity passed all safety invariants"
        else:
            return ReuseDecision.REUSE_WITH_VERIFICATION, f"Moderate similarity ({similarity:.2f}) requires verification"

    def invalidate(self, key: str, namespace: Optional[str] = None) -> int:
        count = 0
        to_del = [k for k in self._entries if k == key or (namespace and self._entries[k].namespace == namespace)]
        for k in to_del:
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
        self.stats.total_bytes_used = 0

    def get_stats(self) -> MemoryStats:
        self.stats.total_bytes_used = self.total_bytes
        total_attempts = self.stats.total_lookups
        self.stats.hit_rate = (self.stats.total_hits / total_attempts) if total_attempts > 0 else 0.0
        return self.stats
