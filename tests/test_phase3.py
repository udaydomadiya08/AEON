"""
Comprehensive Test Suite for AEON Phase 3 — Memory Hierarchy.

Covers all functional requirements, edge cases, benchmarks, and property-style invariants
specified in Sections 34 and 35 of the AEON Phase 3 specification.
"""

import time
import pytest
from aeon import AEON
from aeon.types import (
    InferenceRequest,
    ControllerAction,
    DecisionReason,
    UtilityWeights,
)
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
from aeon.memory.l1_kv_cache import L1KVCache, KVCacheMetadata
from aeon.memory.l2_prefix_cache import L2PrefixCache
from aeon.memory.l3_intermediate_cache import L3IntermediateCache
from aeon.memory.l4_semantic_cache import L4SemanticCache, HybridSemanticMatcher
from aeon.memory.admission import CacheAdmissionController
from aeon.memory.economics import MemoryEconomicsEstimator
from aeon.memory.hierarchy import AEONMemoryHierarchy
from aeon.benchmarks.harness import BenchmarkHarness


# ---------------------------------------------------------------------------
# Core Tier Tests (L1 - L4)
# ---------------------------------------------------------------------------

def test_l1_cache_hit():
    cache = L1KVCache(max_bytes=10 * 1024 * 1024, is_simulated=True)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L1_KV,
        key="What is quantum computing?",
        payload="KV_TENSOR_PTR_0x7f88",
        model_id="aeon-small-1.5b",
        size_bytes=1024,
    )
    assert cache.insert(entry) is True

    req = MemoryRequest(
        query="What is quantum computing?",
        model_id="aeon-small-1.5b",
    )
    res = cache.lookup(req)
    assert res.hit is True
    assert res.level == MemoryLevel.L1_KV
    assert res.entry.payload == "KV_TENSOR_PTR_0x7f88"
    assert "SIMULATED_KV_REUSE" in res.explanation


def test_l1_cache_miss():
    cache = L1KVCache(max_bytes=10 * 1024 * 1024, is_simulated=True)
    req = MemoryRequest(query="Unseen quantum query", model_id="aeon-small-1.5b")
    res = cache.lookup(req)
    assert res.hit is False
    assert res.level == MemoryLevel.L1_KV
    assert res.lookup_cost_usd > 0.0


def test_l2_exact_prefix_hit():
    cache = L2PrefixCache(max_bytes=5 * 1024 * 1024)
    prefix = "You are a helpful assistant for medical triage."
    entry = MemoryEntry(
        memory_level=MemoryLevel.L2_PREFIX,
        key=prefix,
        payload=prefix,
        size_bytes=len(prefix),
    )
    assert cache.insert(entry) is True

    req = MemoryRequest(query=f"{prefix} What are the symptoms of appendicitis?")
    res = cache.lookup(req)
    assert res.hit is True
    assert res.level == MemoryLevel.L2_PREFIX
    assert res.matched_prefix_tokens > 0
    assert res.reuse_ratio > 0.0
    assert res.estimated_compute_saved > 0.0


def test_l2_longest_prefix_match():
    cache = L2PrefixCache(max_bytes=5 * 1024 * 1024)
    short_prefix = "System: Analyze financial reports."
    long_prefix = "System: Analyze financial reports. Follow GAAP conventions and verify quarterly balance sheet."
    
    cache.insert(MemoryEntry(memory_level=MemoryLevel.L2_PREFIX, key=short_prefix, payload=short_prefix, size_bytes=len(short_prefix)))
    cache.insert(MemoryEntry(memory_level=MemoryLevel.L2_PREFIX, key=long_prefix, payload=long_prefix, size_bytes=len(long_prefix)))

    query = f"{long_prefix} Calculate the working capital ratio."
    res = cache.lookup(MemoryRequest(query=query))
    assert res.hit is True
    assert res.entry.key == long_prefix
    assert res.matched_prefix_tokens > len(short_prefix.split())


def test_l3_intermediate_reuse():
    cache = L3IntermediateCache(max_bytes=5 * 1024 * 1024)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L3_INTERMEDIATE,
        key="tokenize('Hello World')",
        namespace="tokenization",
        payload=[15496, 2159],
        size_bytes=64,
        dependencies=["vocab_v1"],
    )
    assert cache.insert(entry) is True

    req = MemoryRequest(query="tokenize('Hello World')", namespace="tokenization")
    res = cache.lookup(req)
    assert res.hit is True
    assert res.level == MemoryLevel.L3_INTERMEDIATE
    assert res.entry.payload == [15496, 2159]


def test_l4_semantic_match():
    cache = L4SemanticCache(max_bytes=5 * 1024 * 1024, similarity_threshold=0.75, high_similarity_threshold=0.85)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L4_SEMANTIC,
        key="What is the capital of France?",
        payload="Paris",
        quality_estimate=0.98,
        confidence=0.95,
    )
    cache.insert(entry)

    req = MemoryRequest(query="What is the capital city of France?")
    res = cache.lookup(req)
    assert res.hit is True
    assert res.level == MemoryLevel.L4_SEMANTIC
    assert res.entry.payload == "Paris"
    assert res.reuse_decision in [ReuseDecision.SAFE_REUSE, ReuseDecision.REUSE_WITH_VERIFICATION]


def test_semantic_false_positive_rejected():
    cache = L4SemanticCache(max_bytes=5 * 1024 * 1024, similarity_threshold=0.55)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L4_SEMANTIC,
        key="What was Apple's total revenue in 2024?",
        payload="$391 Billion",
        quality_estimate=0.95,
        confidence=0.95,
    )
    cache.insert(entry)

    # Similar query but with different year (Section 27 temporal mismatch)
    req = MemoryRequest(query="What will Apple's total revenue be in 2026?")
    res = cache.lookup(req)
    assert res.reuse_decision == ReuseDecision.REJECT_REUSE
    assert "Temporal entity discrepancy" in res.explanation


# ---------------------------------------------------------------------------
# Invalidation, TTL, and Freshness Tests
# ---------------------------------------------------------------------------

def test_ttl_expiration():
    cache = L4SemanticCache(max_bytes=1024 * 1024)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L4_SEMANTIC,
        key="Stock price",
        payload="$100",
        ttl_seconds=0.01,  # 10ms
    )
    cache.insert(entry)
    time.sleep(0.02)  # Wait for expiration

    req = MemoryRequest(query="Stock price")
    res = cache.lookup(req)
    assert res.hit is False


def test_model_version_invalidation():
    cache = L1KVCache(max_bytes=1024 * 1024)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L1_KV,
        key="Test query",
        payload="KV_PTR",
        model_version="1.0.0",
    )
    cache.insert(entry)

    req_compatible = MemoryRequest(query="Test query", model_version="1.0.0")
    assert cache.lookup(req_compatible).hit is True

    req_incompatible = MemoryRequest(query="Test query", model_version="2.0.0")
    res = cache.lookup(req_incompatible)
    assert res.hit is False
    assert "Model/tokenizer version mismatch" in res.explanation


def test_dependency_invalidation():
    cache = L3IntermediateCache(max_bytes=1024 * 1024)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L3_INTERMEDIATE,
        key="parsed_doc_101",
        namespace="extraction",
        payload={"revenue": 5000},
        dependencies=["source_doc_v1", "fx_rates_oct2026"],
    )
    cache.insert(entry)

    assert cache.lookup(MemoryRequest(query="parsed_doc_101", namespace="extraction")).hit is True

    # Invalidate dependency
    invalidated_count = cache.invalidate_by_dependency("source_doc_v1")
    assert invalidated_count == 1
    assert cache.lookup(MemoryRequest(query="parsed_doc_101", namespace="extraction")).hit is False


# ---------------------------------------------------------------------------
# Eviction and Admission Tests
# ---------------------------------------------------------------------------

def test_lru_eviction():
    cache = L4SemanticCache(max_bytes=300, eviction_policy=EvictionPolicy.LRU)
    e1 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Item1", payload="P1", size_bytes=100)
    e2 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Item2", payload="P2", size_bytes=100)
    e3 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Item3", payload="P3", size_bytes=100)

    cache.insert(e1)
    time.sleep(0.005)
    cache.insert(e2)
    time.sleep(0.005)
    cache.insert(e3)

    # Touch Item1 to make Item2 the least recently accessed
    cache.lookup(MemoryRequest(query="Item1"))

    # Inserting e4 should evict Item2
    e4 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Item4", payload="P4", size_bytes=100)
    cache.insert(e4)

    assert "Item2" not in cache._entries
    assert "Item1" in cache._entries
    assert "Item4" in cache._entries


def test_lfu_eviction():
    cache = L4SemanticCache(max_bytes=300, eviction_policy=EvictionPolicy.LFU)
    e1 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Alpha", payload="PA", size_bytes=100, access_count=10)
    e2 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Beta", payload="PB", size_bytes=100, access_count=2)
    e3 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Gamma", payload="PG", size_bytes=100, access_count=5)

    cache.insert(e1)
    cache.insert(e2)
    cache.insert(e3)

    # Insert Delta (100B) -> should evict Beta (least frequently accessed, access_count=2)
    e4 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Delta", payload="PD", size_bytes=100)
    cache.insert(e4)

    assert "Beta" not in cache._entries
    assert "Alpha" in cache._entries


def test_value_aware_eviction():
    cache = L4SemanticCache(max_bytes=300, eviction_policy=EvictionPolicy.VALUE_AWARE)
    # Entry A: small compute savings ($0.000001 per 100B -> value_density = 0.01)
    e1 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="LowVal", payload="P1", size_bytes=100, expected_cost_savings_usd=0.000001)
    e1.calculate_value_density()

    # Entry B: massive compute savings ($0.0005 per 100B -> value_density = 5.0)
    e2 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="HighVal", payload="P2", size_bytes=100, expected_cost_savings_usd=0.0005)
    e2.calculate_value_density()

    # Entry C: moderate savings
    e3 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="MedVal", payload="P3", size_bytes=100, expected_cost_savings_usd=0.0001)
    e3.calculate_value_density()

    cache.insert(e1)
    cache.insert(e2)
    cache.insert(e3)

    # Insert new entry -> should evict LowVal first because of lowest value density
    e4 = MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="NewEntry", payload="P4", size_bytes=100, expected_cost_savings_usd=0.0002)
    e4.calculate_value_density()
    cache.insert(e4)

    assert "LowVal" not in cache._entries
    assert "HighVal" in cache._entries


def test_admission_control():
    controller = CacheAdmissionController(min_net_savings_usd=0.00001)
    
    # 1. Unviable entry (low expected compute savings vs storage cost)
    worthless_entry = MemoryEntry(
        memory_level=MemoryLevel.L4_SEMANTIC,
        key="trivial",
        payload="ok",
        size_bytes=1024,
        expected_cost_savings_usd=0.000001,  # 1e-6 < 1e-5 threshold
    )
    admit, reason = controller.should_admit(worthless_entry, task_type="classification")
    assert admit is False
    assert "below hurdle rate" in reason or "Insufficient" in reason

    # 2. Economically viable entry
    valuable_entry = MemoryEntry(
        memory_level=MemoryLevel.L4_SEMANTIC,
        key="complex reasoning query",
        payload="3-page proof",
        size_bytes=512,
        expected_cost_savings_usd=0.0005,
    )
    admit, _ = controller.should_admit(valuable_entry, task_type="reasoning")
    assert admit is True


def test_memory_budget():
    hierarchy = AEONMemoryHierarchy(budget=MemoryBudget(max_total_bytes=1000))
    req = MemoryRequest(query="Test")
    # Add multiple items exceeding budget
    for i in range(15):
        hierarchy.admit_and_store(
            request=req,
            payload=f"Payload data number {i} " * 5,
            level=MemoryLevel.L4_SEMANTIC,
        )
    assert hierarchy.total_bytes_used <= hierarchy.budget.max_total_bytes


def test_memory_scope_isolation():
    cache = L4SemanticCache(max_bytes=1024 * 1024)
    # Entry scoped to Session A
    entry_a = MemoryEntry(
        memory_level=MemoryLevel.L4_SEMANTIC,
        key="What is my account status?",
        payload="Balance: $500",
        scope=MemoryScope.SESSION,
        scope_id="session-user-123",
    )
    cache.insert(entry_a)

    # Query from Session B should NOT see Session A data
    req_b = MemoryRequest(
        query="What is my account status?",
        scope=MemoryScope.SESSION,
        scope_id="session-user-456",
    )
    assert cache.lookup(req_b).hit is False

    # Query from Session A should succeed
    req_a = MemoryRequest(
        query="What is my account status?",
        scope=MemoryScope.SESSION,
        scope_id="session-user-123",
    )
    assert cache.lookup(req_a).hit is True


def test_memory_cost_model():
    estimator = MemoryEconomicsEstimator(UtilityWeights())
    
    # Hit result
    hit_res = MemoryLookupResult(
        hit=True,
        level=MemoryLevel.L4_SEMANTIC,
        estimated_compute_saved=5e11,
        estimated_cost_saved_usd=0.0001,
        estimated_latency_saved_ms=300.0,
        lookup_cost_usd=1e-7,
        lookup_latency_ms=0.5,
        reuse_decision=ReuseDecision.SAFE_REUSE,
    )
    utility_hit = estimator.evaluate_memory_utility(hit_res, current_difficulty=0.5)
    assert utility_hit > 0.0

    # Miss result must reflect non-zero lookup penalty (Section 26)
    miss_res = MemoryLookupResult(
        hit=False,
        lookup_cost_usd=1e-7,
        lookup_latency_ms=0.5,
    )
    utility_miss = estimator.evaluate_memory_utility(miss_res, current_difficulty=0.5)
    assert utility_miss < 0.0


def test_memory_voi_integration():
    aeon = AEON(policy="voi", memory=True)
    # 1st run: computes and admits to cache
    res1 = aeon.run("What is 12345 + 54321?", memory=True)
    assert res1.trace.cache_hit is False

    # 2nd run: VOI engine detects safe cache reuse as economically superior
    res2 = aeon.run("What is 12345 + 54321?", memory=True)
    assert res2.trace.cache_hit is True
    assert res2.trace.memory_hit is True
    assert res2.trace.stopping_reason == DecisionReason.CACHE_HIT_SAFE.value
    assert res2.compute == 0.0


def test_dynamic_memory_decision():
    aeon = AEON(policy="voi", memory=True)
    prefix = "You are a senior financial analyst. "
    q1 = f"{prefix}Analyze the debt ratio."
    q2 = f"{prefix}Calculate net profit margin."

    aeon.memory.warm_prefix(prefix)
    res = aeon.run(q1, memory=True)
    # L2 prefix cache should trigger partial reuse
    assert res.trace.memory_level in [MemoryLevel.L2_PREFIX.value, MemoryLevel.L4_SEMANTIC.value] or res.trace.reused_tokens > 0


def test_memory_telemetry():
    aeon = AEON(policy="voi", memory=True)
    res = aeon.run("What is the speed of sound?", memory=True)
    t = res.trace
    assert hasattr(t, "memory_hit")
    assert hasattr(t, "memory_level")
    assert hasattr(t, "compute_saved_flops")
    assert hasattr(t, "cost_saved_usd")
    assert hasattr(t, "lookup_latency_ms")


def test_memory_benchmark():
    harness = BenchmarkHarness()
    results = harness.run_phase3()
    assert "1. No Cache" in results
    assert "8. Full L1-L4" in results
    assert "9. Full Memory + AEON VOI" in results
    assert results["9. Full Memory + AEON VOI"]["quality_per_dollar"] > results["1. No Cache"]["quality_per_dollar"]


# ---------------------------------------------------------------------------
# Section 35: Property-Style Invariant Tests
# ---------------------------------------------------------------------------

def test_invariant_1_cache_hit_never_increases_compute():
    """Invariant 1: Cache hit must never increase compute relative to recompute."""
    aeon = AEON(policy="voi", memory=True)
    aeon.run("Compute sqrt(144)", memory=True)
    cached_run = aeon.run("Compute sqrt(144)", memory=True)
    assert cached_run.compute == 0.0


def test_invariant_2_strict_freshness_never_returns_expired():
    """Invariant 2: An expired entry must never be returned under STRICT freshness."""
    cache = L4SemanticCache(max_bytes=1024 * 1024)
    entry = MemoryEntry(
        memory_level=MemoryLevel.L4_SEMANTIC,
        key="Stock price TSLA",
        payload="$220",
        created_at=time.time() - 70.0,  # 70s old
        freshness_policy=FreshnessPolicy.STRICT,
    )
    cache.insert(entry)
    res = cache.lookup(MemoryRequest(query="Stock price TSLA", freshness_policy=FreshnessPolicy.STRICT))
    assert res.reuse_decision == ReuseDecision.REJECT_REUSE


def test_invariant_3_incompatible_model_version_never_reused():
    """Invariant 3: Incompatible model-version must never be reused."""
    cache = L1KVCache(max_bytes=1024 * 1024)
    cache.insert(MemoryEntry(memory_level=MemoryLevel.L1_KV, key="prompt", payload="KV", model_version="1.0.0"))
    res = cache.lookup(MemoryRequest(query="prompt", model_version="2.0.0"))
    assert res.hit is False


def test_invariant_4_memory_usage_never_exceeds_budget():
    """Invariant 4: Memory usage must never exceed budget after eviction completes."""
    budget_bytes = 2000
    cache = L4SemanticCache(max_bytes=budget_bytes)
    for i in range(25):
        cache.insert(MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key=f"k_{i}", payload=f"data_{i}" * 20, size_bytes=200))
    assert cache.total_bytes <= budget_bytes


def test_invariant_5_cross_scope_never_returned():
    """Invariant 5: Cross-scope entries must never be returned."""
    cache = L4SemanticCache(max_bytes=1024 * 1024)
    cache.insert(MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="secret", payload="S1", scope=MemoryScope.USER, scope_id="user_alice"))
    res = cache.lookup(MemoryRequest(query="secret", scope=MemoryScope.USER, scope_id="user_bob"))
    assert res.hit is False


def test_invariant_6_semantic_similarity_alone_not_guarantee_reuse():
    """Invariant 6: High semantic similarity alone must not guarantee reuse without guardrails."""
    cache = L4SemanticCache(max_bytes=1024 * 1024)
    cache.insert(MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="Who was the CEO in 2015?", payload="Person A"))
    res = cache.lookup(MemoryRequest(query="Who is the CEO in 2026?"))
    assert res.reuse_decision != ReuseDecision.SAFE_REUSE


def test_invariant_7_every_memory_decision_has_telemetry():
    """Invariant 7: Every memory decision must have telemetry."""
    aeon = AEON(policy="voi", memory=True)
    res = aeon.run("What is machine learning?", memory=True)
    assert res.trace.memory_level is not None or res.trace.memory_hit is False
    assert res.trace.lookup_latency_ms >= 0.0


def test_invariant_8_every_cache_reuse_decision_has_explanation():
    """Invariant 8: Every cache reuse decision must have an explanation."""
    cache = L4SemanticCache(max_bytes=1024 * 1024)
    cache.insert(MemoryEntry(memory_level=MemoryLevel.L4_SEMANTIC, key="What is 1+1?", payload="2"))
    res = cache.lookup(MemoryRequest(query="What is 1+1?"))
    assert len(res.explanation) > 0
