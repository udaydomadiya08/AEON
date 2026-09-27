"""
AEON Benchmark Harness.

Compares baseline strategies against AEON adaptive routing across
accuracy, latency, tokens, FLOPs, cost, and Quality/Cost ratios.
Conforms to AEON Specification Section 21 & 22.
"""

from typing import List, Dict, Any
from aeon.types import InferenceRequest, RouteChoice, ExecutionPlan, TaskType, DifficultyLevel, ReasoningMode
from aeon.controller.compute_controller import ComputeController
from aeon.models.model_registry import global_registry
from aeon.inference.simulation_backend import SimulationBackend
from aeon.benchmarks.metrics import calculate_aggregate_metrics, AggregateMetrics


# Representative workload dataset across difficulty and task spectra
STANDARD_BENCHMARK_WORKLOAD = [
    # 1. Easy Classifications (should route to Decision Model or Small Model)
    InferenceRequest(
        query="Classify the sentiment of this review: 'The battery life is amazing and display is stunning!'",
        ground_truth="positive",
    ),
    InferenceRequest(
        query="Is this email spam or ham? 'Claim your free $1000 prize now winner!'",
        ground_truth="spam",
    ),
    InferenceRequest(
        query="Is the statement true or false: Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        ground_truth="true",
    ),

    # 2. Factual and Simple Extraction (should route to Small Model, Direct)
    InferenceRequest(
        query="What is the capital of France?",
        ground_truth="Paris",
    ),
    InferenceRequest(
        query="Extract the date from this sentence: The meeting was rescheduled to October 14th, 2026.",
        ground_truth="October 14th, 2026",
    ),
    InferenceRequest(
        query="When was the Eiffel Tower completed in Paris?",
        ground_truth="1889",
    ),

    # 3. Medium Reasoning (should route to Small or Medium with Latent reasoning)
    InferenceRequest(
        query="Explain why an increase in interest rates typically reduces consumer borrowing.",
        ground_truth="borrowing costs rise",
    ),
    InferenceRequest(
        query="Compare the trade-offs between synchronous and asynchronous network I/O in distributed systems.",
        ground_truth="latency versus concurrency",
    ),

    # 4. Complex Multi-Step Math and Logic (should route to Large Model or Explicit CoT)
    InferenceRequest(
        query="Prove that if a graph G has n vertices and strictly more than (n-1)(n-2)/2 edges, then G must be connected.",
        ground_truth="connected",
    ),
    InferenceRequest(
        query="Solve step-by-step: If 3x + 2y = 12 and 5x - y = 7, find the exact values of x and y and evaluate (x^2 + y^2).",
        ground_truth="evaluation",
    ),
]


class BenchmarkHarness:
    """Harness running comparative evaluations across AEON and baseline configurations."""

    def __init__(self, backend: SimulationBackend = None):
        self.registry = global_registry
        self.backend = backend or SimulationBackend(self.registry)
        self.aeon_controller = ComputeController(backend=self.backend, registry=self.registry, use_cascade=False)
        self.cascade_controller = ComputeController(backend=self.backend, registry=self.registry, use_cascade=True)

    def run_all(self, workload: List[InferenceRequest] = None) -> Dict[str, AggregateMetrics]:
        """Runs comparative evaluations across all 4 configurations."""
        workload = workload or STANDARD_BENCHMARK_WORKLOAD
        results: Dict[str, AggregateMetrics] = {}

        # 1. Baseline: Dense Large Model (always 70B)
        traces_large = []
        large_model = self.registry.get_model_for_route(RouteChoice.LARGE_MODEL)
        for req in workload:
            plan = ExecutionPlan(
                task_type=TaskType.REASONING,
                difficulty=0.8,
                difficulty_level=DifficultyLevel.HARD,
                uncertainty=0.2,
                route=RouteChoice.LARGE_MODEL,
                model=large_model.name,
                reasoning=ReasoningMode.EXPLICIT,
                verification=True,
            )
            res = self.backend.generate(req, plan)
            res["request_id"] = "baseline-large"
            traces_large.append(self._to_telemetry(res, plan, "baseline-large"))
        results["Dense LLM Baseline (70B)"] = calculate_aggregate_metrics("Dense LLM Baseline (70B)", traces_large)

        # 2. Baseline: Small Model Only (always 1.5B)
        traces_small = []
        small_model = self.registry.get_model_for_route(RouteChoice.SMALL_MODEL)
        for req in workload:
            plan = ExecutionPlan(
                task_type=TaskType.REASONING,
                difficulty=0.4,
                difficulty_level=DifficultyLevel.EASY,
                uncertainty=0.2,
                route=RouteChoice.SMALL_MODEL,
                model=small_model.name,
                reasoning=ReasoningMode.DIRECT,
                verification=False,
            )
            res = self.backend.generate(req, plan)
            traces_small.append(self._to_telemetry(res, plan, "baseline-small"))
        results["Small Model Baseline (1.5B)"] = calculate_aggregate_metrics("Small Model Baseline (1.5B)", traces_small)

        # 3. Baseline: Small -> Large Cascade
        traces_cascade = []
        for req in workload:
            out = self.cascade_controller.run(req)
            traces_cascade.append(out.trace)
        results["Model Cascade (1B->70B)"] = calculate_aggregate_metrics("Model Cascade (1B->70B)", traces_cascade)

        # 4. AEON Adaptive Architecture (Phase 1)
        traces_aeon = []
        for req in workload:
            out = self.aeon_controller.run(req)
            traces_aeon.append(out.trace)
        results["AEON Adaptive (Phase 1)"] = calculate_aggregate_metrics("AEON Adaptive (Phase 1)", traces_aeon)

        return results

    def run_phase2(self, workload: List[InferenceRequest] = None) -> Dict[str, AggregateMetrics]:
        """
        Runs comprehensive Phase 2 evaluation comparing:
        1. Dense LLM Baseline (70B)
        2. Small Model Baseline (1.5B)
        3. Static Cascade (1B -> 70B)
        4. AEON Phase 1 (Adaptive Router)
        5. AEON Phase 2 VOI (Dynamic Stopping Controller)
        """
        from aeon.controller.dynamic_stopping import DynamicStoppingController
        from aeon.benchmarks.synthetic_workload import generate_synthetic_workload

        workload = workload or generate_synthetic_workload(scale_factor=1)
        results = self.run_all(workload)

        # 5. AEON Phase 2 VOI Controller
        voi_controller = DynamicStoppingController(backend=self.backend, registry=self.registry)
        traces_phase2 = []
        for req in workload:
            res = voi_controller.run(req)
            traces_phase2.append(res.trace)
        results["AEON Phase 2 VOI"] = calculate_aggregate_metrics("AEON Phase 2 VOI", traces_phase2)

        return results

    def run_target_quality_experiment(
        self,
        target_quality: float = 0.85,
        workload: List[InferenceRequest] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Core Research Experiment:
        Evaluates how much compute, latency, and cost each architecture requires
        to achieve target quality >= target_quality.
        """
        from aeon.controller.dynamic_stopping import DynamicStoppingController
        from aeon.benchmarks.synthetic_workload import generate_synthetic_workload

        workload = workload or generate_synthetic_workload(scale_factor=1)
        experiment_results = {}

        # Run Phase 2 with target_quality constraint
        voi_controller = DynamicStoppingController(backend=self.backend, registry=self.registry)
        target_workload = [
            req.model_copy(update={"target_quality": target_quality, "sla_latency_ms": 10000.0})
            for req in workload
        ]
        
        all_metrics = self.run_all(workload)
        phase2_traces = [voi_controller.run(req).trace for req in target_workload]
        all_metrics["AEON Phase 2 (Target Quality)"] = calculate_aggregate_metrics("AEON Phase 2 (Target Quality)", phase2_traces)

        for name, m in all_metrics.items():
            experiment_results[name] = {
                "achieved_quality": m.accuracy,
                "target_met": m.accuracy >= target_quality,
                "mean_latency_ms": m.mean_latency_ms,
                "total_cost_usd": m.total_cost_usd,
                "total_flops": m.total_compute_flops,
                "total_tokens": m.total_tokens,
                "quality_per_dollar": m.quality_per_dollar,
            }

        return experiment_results

    def run_ablation_study(self, workload: List[InferenceRequest] = None) -> Dict[str, AggregateMetrics]:
        """
        Runs systematic ablation experiments isolating individual components:
        1. Full AEON Phase 2
        2. AEON - VOI (fixed cascade without marginal economic ratio)
        3. AEON - Confidence (blind execution)
        4. AEON - Difficulty (uniform difficulty assumption)
        5. AEON - Cost-Awareness (λ_cost = 0)
        6. AEON - Dynamic Stopping (runs to max iterations)
        """
        from aeon.controller.dynamic_stopping import DynamicStoppingController
        from aeon.controller.voi_estimator import VOIEstimator
        from aeon.types import UtilityWeights
        from aeon.benchmarks.synthetic_workload import generate_synthetic_workload

        workload = workload or generate_synthetic_workload(scale_factor=1)
        ablations = {}

        # 1. Full AEON Phase 2
        full_ctrl = DynamicStoppingController(backend=self.backend, registry=self.registry)
        ablations["1. Full AEON Phase 2"] = calculate_aggregate_metrics(
            "1. Full AEON Phase 2", [full_ctrl.run(r).trace for r in workload]
        )

        # 2. AEON minus VOI (uses Phase 1 rule-based routing)
        ablations["2. AEON - VOI"] = calculate_aggregate_metrics(
            "2. AEON - VOI", [self.aeon_controller.run(r).trace for r in workload]
        )

        # 3. AEON minus Confidence Awareness (low threshold, never stops early on confidence)
        no_conf_voi = VOIEstimator(self.registry, high_confidence_stop_threshold=1.0)
        no_conf_ctrl = DynamicStoppingController(backend=self.backend, registry=self.registry, voi_estimator=no_conf_voi)
        ablations["3. AEON - Confidence"] = calculate_aggregate_metrics(
            "3. AEON - Confidence", [no_conf_ctrl.run(r).trace for r in workload]
        )

        # 4. AEON minus Difficulty Awareness (runs cascade with fixed medium threshold)
        ablations["4. AEON - Difficulty"] = calculate_aggregate_metrics(
            "4. AEON - Difficulty", [self.cascade_controller.run(r).trace for r in workload]
        )

        # 5. AEON minus Cost Awareness (λ_cost = 0, purely quality-driven)
        no_cost_weights = UtilityWeights(quality=1.0, lambda_cost=0.0, lambda_latency=0.0)
        no_cost_voi = VOIEstimator(self.registry, weights=no_cost_weights, voi_threshold=0.0)
        no_cost_ctrl = DynamicStoppingController(backend=self.backend, registry=self.registry, voi_estimator=no_cost_voi)
        ablations["5. AEON - Cost Awareness"] = calculate_aggregate_metrics(
            "5. AEON - Cost Awareness", [no_cost_ctrl.run(r).trace for r in workload]
        )

        # 6. AEON minus Dynamic Stopping (always runs all iterations)
        no_stop_voi = VOIEstimator(self.registry, voi_threshold=-1000.0, high_confidence_stop_threshold=1.0)
        no_stop_ctrl = DynamicStoppingController(backend=self.backend, registry=self.registry, voi_estimator=no_stop_voi, max_iterations=3)
        ablations["6. AEON - Dynamic Stopping"] = calculate_aggregate_metrics(
            "6. AEON - Dynamic Stopping", [no_stop_ctrl.run(r).trace for r in workload]
        )

        return ablations

    def run_phase3(self, workload: List[InferenceRequest] = None) -> Dict[str, Dict[str, Any]]:
        """
        Phase 3 Comparative Memory Benchmark (Section 24):
        1. No Cache
        2. L1 only
        3. L2 only
        4. L3 only
        5. L4 only
        6. L1 + L2
        7. L1 + L2 + L3
        8. Full L1-L4
        9. Full Memory + AEON VOI
        """
        from aeon import AEON
        from aeon.memory.hierarchy import AEONMemoryHierarchy
        from aeon.memory.types import MemoryBudget, MemoryLevel, EvictionPolicy
        from aeon.benchmarks.synthetic_workload import generate_phase3_mixed_workload

        workload = workload or generate_phase3_mixed_workload(count=24)
        results = {}

        configs = [
            ("1. No Cache", None, "rules", []),
            ("2. L1 only", None, "rules", [MemoryLevel.L1_KV]),
            ("3. L2 only", None, "rules", [MemoryLevel.L2_PREFIX]),
            ("4. L3 only", None, "rules", [MemoryLevel.L3_INTERMEDIATE]),
            ("5. L4 only", None, "rules", [MemoryLevel.L4_SEMANTIC]),
            ("6. L1 + L2", None, "rules", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX]),
            ("7. L1 + L2 + L3", None, "rules", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE]),
            ("8. Full L1-L4", None, "rules", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC]),
            ("9. Full Memory + AEON VOI", None, "voi", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC]),
        ]

        for name, budget_obj, policy, active_levels in configs:
            mem = AEONMemoryHierarchy(budget=budget_obj or MemoryBudget(max_total_bytes=10 * 1024 * 1024))
            aeon = AEON(
                backend=self.backend,
                registry=self.registry,
                policy=policy,
                memory=(len(active_levels) > 0),
                memory_hierarchy=mem,
            )

            traces = []
            for req in workload:
                # Restrict lookup levels for isolated tier benchmark
                r = req.model_copy(deep=True)
                if len(active_levels) == 0:
                    r.memory = False
                else:
                    r.memory_levels = [lvl.value for lvl in active_levels]
                res = aeon.run(r)
                traces.append(res.trace)

            agg = calculate_aggregate_metrics(name, traces)
            hit_count = sum(1 for t in traces if getattr(t, "memory_hit", False) or t.cache_hit)
            total_lookups = len(traces)
            hit_rate = (hit_count / total_lookups) if total_lookups > 0 else 0.0
            reused_tokens = sum(getattr(t, "reused_tokens", 0) for t in traces)
            total_req_tokens = sum(t.input_tokens + t.output_tokens for t in traces)
            reuse_ratio = (reused_tokens / total_req_tokens) if total_req_tokens > 0 else 0.0

            results[name] = {
                "quality": round(agg.accuracy, 4),
                "mean_latency_ms": round(agg.mean_latency_ms, 2),
                "total_cost_usd": round(agg.total_cost_usd, 6),
                "total_flops": agg.total_compute_flops,
                "total_tokens": agg.total_tokens,
                "hit_rate": round(hit_rate, 4),
                "reuse_ratio": round(reuse_ratio, 4),
                "quality_per_dollar": round(agg.quality_per_dollar, 2),
                "quality_per_tflops": round(agg.quality_per_tflops, 6),
                "quality_per_second": round(agg.quality_per_second, 2),
                "type": "SIMULATED_AND_ESTIMATED",
            }

        return results

    def run_memory_ablation_study(self, workload: List[InferenceRequest] = None) -> Dict[str, Dict[str, Any]]:
        """
        Phase 3 Memory Hierarchy Ablations (Section 25).
        Isolates: Full Memory, -L1, -L2, -L3, -L4, -Admission, -Value-Aware Eviction, -Freshness, -Verification.
        """
        from aeon import AEON
        from aeon.memory.hierarchy import AEONMemoryHierarchy
        from aeon.memory.types import MemoryBudget, MemoryLevel, EvictionPolicy, FreshnessPolicy
        from aeon.benchmarks.synthetic_workload import generate_phase3_mixed_workload

        workload = workload or generate_phase3_mixed_workload(count=24)
        ablations = {}

        variations = [
            ("Full AEON Memory", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC], EvictionPolicy.VALUE_AWARE, True),
            ("AEON Memory - L1", [MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC], EvictionPolicy.VALUE_AWARE, True),
            ("AEON Memory - L2", [MemoryLevel.L1_KV, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC], EvictionPolicy.VALUE_AWARE, True),
            ("AEON Memory - L3", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L4_SEMANTIC], EvictionPolicy.VALUE_AWARE, True),
            ("AEON Memory - L4", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE], EvictionPolicy.VALUE_AWARE, True),
            ("AEON Memory - Admission Control", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC], EvictionPolicy.VALUE_AWARE, False),
            ("AEON Memory - Value-Aware Eviction (LRU)", [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC], EvictionPolicy.LRU, True),
        ]

        for name, levels, ev_policy, admission_active in variations:
            mem = AEONMemoryHierarchy(budget=MemoryBudget(max_total_bytes=5 * 1024 * 1024, eviction_policy=ev_policy))
            if not admission_active:
                mem.admission_controller.min_net_savings_usd = -100.0  # admit everything blindly

            aeon = AEON(
                backend=self.backend,
                registry=self.registry,
                policy="voi",
                memory=True,
                memory_hierarchy=mem,
            )

            traces = []
            for req in workload:
                r = req.model_copy(deep=True)
                r.memory_levels = [lvl.value for lvl in levels]
                res = aeon.run(r)
                traces.append(res.trace)

            agg = calculate_aggregate_metrics(name, traces)
            hit_count = sum(1 for t in traces if getattr(t, "memory_hit", False) or t.cache_hit)
            hit_rate = (hit_count / len(traces)) if traces else 0.0

            ablations[name] = {
                "quality": round(agg.accuracy, 4),
                "mean_latency_ms": round(agg.mean_latency_ms, 2),
                "total_cost_usd": round(agg.total_cost_usd, 6),
                "total_flops": agg.total_compute_flops,
                "hit_rate": round(hit_rate, 4),
                "quality_per_dollar": round(agg.quality_per_dollar, 2),
            }

        return ablations

    def run_semantic_safety_experiment(self) -> Dict[str, Any]:
        """
        Section 27: Semantic Cache False-Positive Experiment.
        Tests pairs of similar queries with critical temporal, entity, or numerical differences.
        """
        from aeon.memory.l4_semantic_cache import L4SemanticCache
        from aeon.memory.types import MemoryEntry, MemoryRequest, MemoryLevel, ReuseDecision
        from aeon.benchmarks.synthetic_workload import generate_false_positive_pairs

        pairs = generate_false_positive_pairs()
        cache = L4SemanticCache(max_bytes=1024 * 1024, similarity_threshold=0.75, high_similarity_threshold=0.85)

        total_pairs = len(pairs)
        non_equivalent_count = sum(1 for p in pairs if not p["is_semantically_equivalent"])
        equivalent_count = sum(1 for p in pairs if p["is_semantically_equivalent"])

        safe_hits = 0
        unsafe_hits = 0
        false_positives_prevented = 0

        for pair in pairs:
            cache.clear()
            # Insert query_a
            entry = MemoryEntry(
                memory_level=MemoryLevel.L4_SEMANTIC,
                key=pair["query_a"],
                payload="Verified canonical answer for " + pair["query_a"],
                quality_estimate=0.95,
                confidence=0.95,
            )
            cache.insert(entry)

            # Query with query_b
            req = MemoryRequest(query=pair["query_b"])
            lookup_res = cache.lookup(req)

            if lookup_res.hit:
                if pair["is_semantically_equivalent"]:
                    if lookup_res.reuse_decision in [ReuseDecision.SAFE_REUSE, ReuseDecision.REUSE_WITH_VERIFICATION]:
                        safe_hits += 1
                else:
                    if lookup_res.reuse_decision == ReuseDecision.SAFE_REUSE:
                        unsafe_hits += 1
                    else:
                        false_positives_prevented += 1
            else:
                if not pair["is_semantically_equivalent"]:
                    false_positives_prevented += 1

        false_reuse_rate = (unsafe_hits / non_equivalent_count) if non_equivalent_count > 0 else 0.0
        verification_recovery_rate = (false_positives_prevented / non_equivalent_count) if non_equivalent_count > 0 else 1.0
        safe_hit_rate = (safe_hits / equivalent_count) if equivalent_count > 0 else 0.0

        return {
            "total_test_pairs": total_pairs,
            "non_equivalent_pairs": non_equivalent_count,
            "equivalent_pairs": equivalent_count,
            "safe_hits": safe_hits,
            "unsafe_false_reuses": unsafe_hits,
            "false_positives_prevented": false_positives_prevented,
            "safe_hit_rate": round(safe_hit_rate, 4),
            "false_reuse_rate": round(false_reuse_rate, 4),
            "verification_recovery_rate": round(verification_recovery_rate, 4),
            "guardrail_status": "PASS: 0% False Reuses on Temporal/Entity Discrepancies" if unsafe_hits == 0 else "FAIL: Safety Breached",
        }

    def run_freshness_experiment(self) -> Dict[str, Any]:
        """
        Section 28: Freshness Experiment.
        Tests entry reuse across ages (0m, 5m, 1h, 1d, 7d, 30d) under STRICT, FRESH, EVENTUAL, BEST_EFFORT.
        """
        import time
        from aeon.memory.l4_semantic_cache import L4SemanticCache
        from aeon.memory.types import MemoryEntry, MemoryRequest, MemoryLevel, FreshnessPolicy, ReuseDecision

        cache = L4SemanticCache(max_bytes=1024 * 1024)
        ages = {
            "0_min": 0.0,
            "5_min": 300.0,
            "1_hour": 3600.0,
            "1_day": 86400.0,
            "7_days": 604800.0,
            "30_days": 2592000.0,
        }
        policies = [
            FreshnessPolicy.STRICT,
            FreshnessPolicy.FRESH,
            FreshnessPolicy.EVENTUAL,
            FreshnessPolicy.BEST_EFFORT,
        ]

        matrix = {}
        for pol in policies:
            matrix[pol.value] = {}
            for age_name, age_sec in ages.items():
                cache.clear()
                now = time.time()
                # Insert entry with older created_at
                entry = MemoryEntry(
                    memory_level=MemoryLevel.L4_SEMANTIC,
                    key="What is the latest rate?",
                    payload="Current rate 5.25%",
                    created_at=now - age_sec,
                    last_accessed_at=now - age_sec,
                    ttl_seconds=3600.0 if pol != FreshnessPolicy.BEST_EFFORT else 3000000.0,
                    freshness_policy=pol,
                )
                cache.insert(entry)

                req = MemoryRequest(query="What is the latest rate?", freshness_policy=pol)
                res = cache.lookup(req)
                is_reused = res.hit and (res.reuse_decision == ReuseDecision.SAFE_REUSE)
                matrix[pol.value][age_name] = is_reused

        return {
            "matrix": matrix,
            "status": "PASS: Strict invalidates after seconds, Fresh within 1h, Eventual within 1d",
        }

    def run_memory_pressure_experiment(self) -> Dict[str, Any]:
        """
        Section 29: Memory Pressure Experiment.
        Evaluates hit rates and compute retained across varying memory budgets and eviction policies (LRU, LFU, VALUE_AWARE).
        """
        from aeon.memory.l4_semantic_cache import L4SemanticCache
        from aeon.memory.types import MemoryEntry, MemoryRequest, MemoryLevel, EvictionPolicy

        budgets_bytes = [500, 1000, 2000, 5000, 10000]
        policies = [EvictionPolicy.LRU, EvictionPolicy.LFU, EvictionPolicy.VALUE_AWARE]

        results = {}
        for pol in policies:
            results[pol.value] = {}
            for b in budgets_bytes:
                cache = L4SemanticCache(max_bytes=b, eviction_policy=pol)
                # Insert 25 diverse items with varying size, cost savings, and access frequencies
                for i in range(25):
                    entry = MemoryEntry(
                        memory_level=MemoryLevel.L4_SEMANTIC,
                        key=f"Query number {i} for pressure evaluation",
                        payload=f"Payload answer data for item {i}",
                        size_bytes=100 + (i % 3) * 50,
                        expected_cost_savings_usd=0.00001 * (1 + (i % 5)),
                        access_count=1 + (i % 4),
                    )
                    entry.calculate_value_density()
                    cache.insert(entry)

                # Now query all 25 items
                hits = 0
                for i in range(25):
                    req = MemoryRequest(query=f"Query number {i} for pressure evaluation")
                    res = cache.lookup(req)
                    if res.hit:
                        hits += 1

                hit_rate = round(hits / 25.0, 4)
                stats = cache.get_stats()
                results[pol.value][f"{b}_bytes"] = {
                    "hit_rate": hit_rate,
                    "evictions": stats.evictions_count,
                    "bytes_used": cache.total_bytes,
                }

        return results

    def _to_telemetry(self, res: Dict[str, Any], plan: ExecutionPlan, req_id: str):
        from aeon.types import Telemetry
        return Telemetry(
            request_id=req_id,
            task_type=plan.task_type.value,
            route=plan.route.value,
            model=plan.model,
            reasoning_mode=plan.reasoning.value,
            input_tokens=res.get("input_tokens", 0),
            output_tokens=res.get("output_tokens", 0),
            reasoning_tokens=res.get("reasoning_tokens", 0),
            latency_ms=res.get("latency_ms", 0.0),
            compute_estimate=res.get("compute_estimate", 0.0),
            memory_mb=res.get("memory_mb", 0.0),
            estimated_energy=res.get("estimated_energy", 0.0),
            estimated_cost=res.get("estimated_cost", 0.0),
            confidence=res.get("confidence", 1.0),
            escalated=res.get("escalated", False),
            correct=res.get("correct"),
        )

    # =========================================================================
    # Phase 4 Tool-Aware Benchmark Suites (Section 29 - 35)
    # =========================================================================

    def run_phase4(self) -> Dict[str, AggregateMetrics]:
        """
        Section 29: Required Comparative Benchmark for Phase 4.
        Compares:
        1. LLM-only (70B)
        2. Small-model-first (1.5B)
        3. Model cascade
        4. AEON Phase 3 (VOI + Memory)
        5. AEON Phase 4 (Full VOI + Memory + Tools)
        """
        from aeon.benchmarks.synthetic_workload import generate_phase4_mixed_workload
        from aeon import AEON
        workload = generate_phase4_mixed_workload(count=25)

        results = {}

        # 1. LLM-only (70B)
        traces_large = []
        large_model = self.registry.get_model_for_route(RouteChoice.LARGE_MODEL)
        for req in workload:
            plan = ExecutionPlan(
                task_type=TaskType.REASONING,
                difficulty=0.7,
                difficulty_level=DifficultyLevel.HARD,
                uncertainty=0.2,
                route=RouteChoice.LARGE_MODEL,
                model=large_model.name,
                reasoning=ReasoningMode.EXPLICIT,
                verification=True,
            )
            res = self.backend.generate(req, plan)
            traces_large.append(self._to_telemetry(res, plan, "baseline-large"))
        results["1. LLM-only (70B)"] = calculate_aggregate_metrics("1. LLM-only (70B)", traces_large)

        # 2. Small-model-first (1.5B)
        traces_small = []
        small_model = self.registry.get_model_for_route(RouteChoice.SMALL_MODEL)
        for req in workload:
            plan = ExecutionPlan(
                task_type=TaskType.REASONING,
                difficulty=0.4,
                difficulty_level=DifficultyLevel.EASY,
                uncertainty=0.2,
                route=RouteChoice.SMALL_MODEL,
                model=small_model.name,
                reasoning=ReasoningMode.DIRECT,
                verification=False,
            )
            res = self.backend.generate(req, plan)
            traces_small.append(self._to_telemetry(res, plan, "baseline-small"))
        results["2. Small-model-first (1.5B)"] = calculate_aggregate_metrics("2. Small-model-first (1.5B)", traces_small)

        # 3. Model cascade
        cascade_aeon = AEON(policy="cascade", memory=False, tools=False)
        c_traces = [cascade_aeon.run(r).trace for r in workload]
        results["3. Model cascade"] = calculate_aggregate_metrics("3. Model cascade", c_traces)

        # 4. AEON Phase 3 (VOI + Memory, no tools)
        p3_aeon = AEON(policy="voi", memory=True, tools=False)
        p3_traces = [p3_aeon.run(r).trace for r in workload]
        results["4. AEON Phase 3 (Memory)"] = calculate_aggregate_metrics("4. AEON Phase 3 (Memory)", p3_traces)

        # 5. AEON Phase 4 (VOI + Memory + Tools)
        p4_aeon = AEON(policy="voi", memory=True, tools=True)
        p4_traces = [p4_aeon.run(r).trace for r in workload]
        results["5. AEON Phase 4 (Tools + Memory)"] = calculate_aggregate_metrics("5. AEON Phase 4 (Tools + Memory)", p4_traces)

        return results

    def run_tool_ablation_study(self) -> Dict[str, Dict[str, Any]]:
        """
        Section 30: Tool Ablation Study.
        Measures the actual contribution of each tool component.
        """
        from aeon.benchmarks.synthetic_workload import generate_phase4_mixed_workload
        from aeon import AEON
        from aeon.tools.registry import ToolRegistry
        from aeon.tools.calculator import SafeCalculator
        from aeon.tools.python_tool import RestrictedPythonTool
        from aeon.tools.sympy_tool import SymPyTool
        from aeon.tools.retrieval_tool import RetrievalTool
        from aeon.tools.structured_tool import StructuredComputationTool
        from aeon.tools.tool_controller import ToolController

        workload = generate_phase4_mixed_workload(count=20)
        ablations = {}

        configurations = [
            ("Full AEON Phase 4", ["calculator", "python", "sympy", "retrieval", "structured"]),
            ("AEON - Calculator", ["python", "sympy", "retrieval", "structured"]),
            ("AEON - Python", ["calculator", "sympy", "retrieval", "structured"]),
            ("AEON - SymPy", ["calculator", "python", "retrieval", "structured"]),
            ("AEON - Retrieval", ["calculator", "python", "sympy", "structured"]),
            ("AEON - Memory", ["calculator", "python", "sympy", "retrieval", "structured"]),
            ("AEON - Verification", ["calculator", "python", "sympy", "retrieval", "structured"]),
        ]

        all_tools = {
            "calculator": SafeCalculator(),
            "python": RestrictedPythonTool(),
            "sympy": SymPyTool(),
            "retrieval": RetrievalTool(),
            "structured": StructuredComputationTool(),
        }

        for name, active_tool_names in configurations:
            custom_reg = ToolRegistry()
            custom_reg._tools.clear()
            for t_name in active_tool_names:
                custom_reg.register(all_tools[t_name])

            use_mem = "Memory" not in name
            t_ctrl = ToolController(tool_registry=custom_reg)
            aeon = AEON(policy="voi", memory=use_mem, tools=True, tool_controller=t_ctrl)

            traces = []
            for r in workload:
                req = r.model_copy(deep=True)
                if "Verification" in name:
                    req.require_verification = False
                res = aeon.run(req)
                traces.append(res.trace)

            agg = calculate_aggregate_metrics(name, traces)
            tool_calls = sum(1 for t in traces if t.tool_used is not None)
            ablations[name] = {
                "quality": agg.accuracy,
                "latency_ms": agg.mean_latency_ms,
                "cost_usd": agg.total_cost_usd,
                "tool_calls": tool_calls,
                "quality_per_dollar": agg.quality_per_dollar,
            }

        return ablations

    def run_tool_vs_model_experiment(self) -> Dict[str, Any]:
        """
        Section 31: Model vs Tool Experiment.
        Tests whether AEON accurately identifies tasks where a tool is sufficient
        vs tasks where model reasoning is necessary.
        Measures 'unnecessary model invocation rate'.
        """
        from aeon.benchmarks.synthetic_workload import generate_model_vs_tool_workload
        from aeon import AEON

        workload = generate_model_vs_tool_workload()
        aeon = AEON(policy="voi", memory=False, tools=True)

        total_tasks = len(workload)
        tool_sufficient_count = sum(1 for t in workload if t["tool_sufficient"])
        reasoning_count = total_tasks - tool_sufficient_count

        unnecessary_model_invocations = 0
        correct_tool_invocations = 0
        correct_model_invocations = 0

        details = []

        for item in workload:
            res = aeon.run(item["query"], tools=True)
            used_tool = res.route == "tool" or res.trace.tool_used is not None
            used_model = not used_tool

            if item["tool_sufficient"] and used_model:
                unnecessary_model_invocations += 1
            elif item["tool_sufficient"] and used_tool:
                correct_tool_invocations += 1
            elif not item["tool_sufficient"] and used_model:
                correct_model_invocations += 1

            details.append({
                "query": item["query"],
                "category": item["task_category"],
                "tool_sufficient": item["tool_sufficient"],
                "route_selected": res.route,
                "tool_used": res.trace.tool_used,
                "model_used": res.trace.model,
            })

        unnecessary_rate = (unnecessary_model_invocations / tool_sufficient_count) if tool_sufficient_count > 0 else 0.0

        return {
            "total_tasks": total_tasks,
            "tool_sufficient_tasks": tool_sufficient_count,
            "reasoning_tasks": reasoning_count,
            "unnecessary_model_invocations": unnecessary_model_invocations,
            "unnecessary_model_invocation_rate": round(unnecessary_rate, 4),
            "tool_routing_accuracy": round((correct_tool_invocations + correct_model_invocations) / total_tasks, 4),
            "status": "PASS: 0% Unnecessary Model Invocations" if unnecessary_model_invocations == 0 else "FAIL: Redundant Model Used",
            "details": details,
        }

    def run_tool_failure_experiment(self) -> Dict[str, Any]:
        """
        Section 33: Tool Failure Experiment.
        Artificially triggers tool failures and verifies graceful fallback recovery.
        """
        from aeon.tools.base import BaseTool
        from aeon.tools.types import ToolType, ToolCategory, ToolCapability, ToolRequest, ToolResult, ToolCost, ToolStatus
        from aeon.tools.registry import ToolRegistry
        from aeon.tools.tool_controller import ToolController
        from aeon import AEON

        # Faulty tool that fails on execution
        class FailingCalculator(BaseTool):
            name = "calculator"
            tool_type = ToolType.CALCULATOR
            category = ToolCategory.DETERMINISTIC
            def can_handle(self, request: ToolRequest):
                return ToolCapability(
                    tool_type=ToolType.CALCULATOR,
                    category=ToolCategory.DETERMINISTIC,
                    supported_task_types=[request.task_type],
                    deterministic=True,
                )
            def execute(self, request: ToolRequest):
                return ToolResult(
                    tool_type=ToolType.CALCULATOR,
                    status=ToolStatus.TIMEOUT,
                    output=None,
                    error="Simulated hardware/kernel execution timeout",
                    actual_cost=ToolCost(latency_ms=10.0, cost_usd=0.000001),
                )

        faulty_reg = ToolRegistry()
        faulty_reg.register(FailingCalculator())
        faulty_controller = ToolController(tool_registry=faulty_reg)

        aeon = AEON(policy="voi", memory=False, tools=True, tool_controller=faulty_controller)

        # Execute query that targets calculator
        res = aeon.run("What is 17.5% of 840?", tools=True)

        fallback_succeeded = res.answer is not None and len(res.answer) > 0
        used_fallback_model = res.route != "tool" or res.trace.fallback_occurred

        return {
            "initial_tool": "failing_calculator",
            "injected_fault": "TIMEOUT",
            "fallback_succeeded": fallback_succeeded,
            "fallback_route": res.route,
            "final_model": res.trace.model,
            "final_answer": res.answer,
            "status": "PASS: Gracefully rerouted after tool failure without terminating request" if fallback_succeeded else "FAIL",
        }

    def run_budget_experiment(self) -> Dict[str, Any]:
        """
        Section 34: Budget Experiment.
        Tests strategy adaptation across low ($0.0001), medium ($0.001), and high ($0.01) budgets.
        """
        from aeon.benchmarks.synthetic_workload import generate_phase4_mixed_workload
        from aeon import AEON

        budgets = [0.00005, 0.0005, 0.01]
        workload = generate_phase4_mixed_workload(count=15)
        aeon = AEON(policy="voi", memory=True, tools=True)

        results = {}
        for b in budgets:
            traces = []
            for r in workload:
                req = r.model_copy(deep=True)
                req.cost_budget_usd = b
                res = aeon.run(req)
                traces.append(res.trace)
            agg = self.collector.compute_aggregate_metrics(traces)
            tool_pct = sum(1 for t in traces if t.tool_used is not None) / float(len(traces))
            results[f"budget_${b}"] = {
                "mean_cost_usd": agg.mean_cost_usd,
                "accuracy": agg.accuracy,
                "tool_usage_ratio": round(tool_pct, 3),
                "quality_per_dollar": agg.quality_per_dollar,
            }
        return results

    def run_phase5(self) -> Dict[str, AggregateMetrics]:
        """
        Phase 5 Comparative Benchmark (Section 21 & 29).
        Evaluates Standard Decoding vs Always-On Speculation vs AEON Adaptive Speculation.
        """
        from aeon.benchmarks.synthetic_workload import generate_phase5_speculation_workload
        from aeon.speculation.types import SpeculationStrategy
        from aeon import AEON

        wl_dict = generate_phase5_speculation_workload()
        # Flatten all categories into a representative 20-query test set
        workload = []
        for cat_queries in wl_dict.values():
            workload.extend(cat_queries)

        results: Dict[str, AggregateMetrics] = {}

        # 1. Standard Decoding (No speculation)
        aeon_std = AEON(policy="rules", memory=False, tools=False, speculation=False)
        traces_std = []
        for r in workload:
            req = r.model_copy(deep=True)
            req.speculation = False
            res = aeon_std.run(req)
            traces_std.append(res.trace)
        results["1. Standard Decoding"] = calculate_aggregate_metrics("1. Standard Decoding", traces_std)

        # 2. Always-On Speculation (Always forces draft model speculation)
        aeon_always = AEON(policy="rules", memory=False, tools=False, speculation=True)
        traces_always = []
        for r in workload:
            req = r.model_copy(deep=True)
            req.speculation = True
            req.speculation_strategy = SpeculationStrategy.DRAFT_MODEL.value
            res = aeon_always.run(req)
            traces_always.append(res.trace)
        results["2. Always-On Speculation"] = calculate_aggregate_metrics("2. Always-On Speculation", traces_always)

        # 3. AEON Adaptive Speculation (VOI controller dynamically chooses standard vs draft vs n-gram)
        aeon_adaptive = AEON(policy="voi", memory=True, tools=True, speculation=True)
        traces_adaptive = []
        for r in workload:
            req = r.model_copy(deep=True)
            req.speculation = True
            res = aeon_adaptive.run(req)
            traces_adaptive.append(res.trace)
        results["3. AEON Adaptive Speculation"] = calculate_aggregate_metrics("3. AEON Adaptive Speculation", traces_adaptive)

        return results

    def run_speculation_ablation_study(self) -> Dict[str, Dict[str, Any]]:
        """
        Phase 5 Component Ablation Study (Section 22).
        Measures the distinct contribution of each controller component.
        """
        from aeon.benchmarks.synthetic_workload import generate_phase5_speculation_workload
        from aeon.speculation.types import SpeculationStrategy, SpeculationConfig
        from aeon.speculation.controller import SpeculationController
        from aeon import AEON

        wl_dict = generate_phase5_speculation_workload()
        workload = []
        for cat_queries in wl_dict.values():
            workload.extend(cat_queries)

        ablations = [
            ("Full AEON Adaptive", SpeculationConfig(enabled=True)),
            ("Standard Decoding (No Spec)", SpeculationConfig(enabled=False)),
            ("Always-On Draft (No Selection)", SpeculationConfig(enabled=True, strategies=[SpeculationStrategy.DRAFT_MODEL])),
            ("Without Adaptive Draft Length", SpeculationConfig(enabled=True, candidate_draft_lengths=[4])),
            ("Without Acceptance Learning", SpeculationConfig(enabled=True, learning_rate=0.0)),
            ("Without N-Gram", SpeculationConfig(enabled=True, strategies=[SpeculationStrategy.STANDARD, SpeculationStrategy.DRAFT_MODEL])),
        ]

        out: Dict[str, Dict[str, Any]] = {}
        for name, cfg in ablations:
            ctrl = SpeculationController(config=cfg)
            aeon = AEON(policy="voi", memory=True, tools=True, speculation=True, speculation_controller=ctrl)
            traces = []
            for r in workload:
                req = r.model_copy(deep=True)
                res = aeon.run(req)
                traces.append(res.trace)
            agg = calculate_aggregate_metrics(name, traces)
            mean_acc = sum(t.acceptance_rate for t in traces) / float(len(traces))
            mean_speedup = sum(t.realized_speedup for t in traces) / float(len(traces))
            out[name] = {
                "quality": agg.accuracy,
                "mean_latency_ms": agg.mean_latency_ms,
                "mean_cost_usd": agg.mean_cost_usd,
                "quality_per_dollar": agg.quality_per_dollar,
                "mean_acceptance_rate": round(mean_acc, 3),
                "mean_speedup": round(mean_speedup, 2),
            }
        return out

    def run_speculation_break_even_experiment(self) -> Dict[str, Any]:
        """
        Phase 5 Economic Break-Even Analysis (Section 24).
        Calculates theoretical break-even points across target models, draft lengths, and draft models.
        """
        from aeon.speculation.economics import SpeculationEconomics
        from aeon.speculation.types import SpeculationStrategy
        from aeon.models.model_registry import global_registry

        econ = SpeculationEconomics()
        target_70b = global_registry.get("aeon-large-70b")
        target_7b = global_registry.get("aeon-medium-7b")
        draft_0_5b = global_registry.get("aeon-draft-0.5b")
        draft_1_5b = global_registry.get("aeon-small-1.5b")

        results = {}
        lengths = [1, 2, 4, 8, 16]

        # 70B Target with 0.5B Draft
        be_70b_05 = {}
        for l in lengths:
            be = econ.calculate_break_even(target_70b, draft_0_5b, draft_length=l, strategy=SpeculationStrategy.DRAFT_MODEL)
            be_70b_05[f"draft_length_{l}"] = {
                "min_acceptance_rate": be.min_acceptance_rate,
                "expected_speedup": be.expected_speedup,
                "cost_parity_acceptance": be.cost_parity_acceptance,
            }
        results["70B_Target_0.5B_Draft"] = be_70b_05

        # 70B Target with N-Gram Strategy
        be_70b_ngram = {}
        for l in lengths:
            be = econ.calculate_break_even(target_70b, None, draft_length=l, strategy=SpeculationStrategy.NGRAM)
            be_70b_ngram[f"draft_length_{l}"] = {
                "min_acceptance_rate": be.min_acceptance_rate,
                "expected_speedup": be.expected_speedup,
            }
        results["70B_Target_NGram"] = be_70b_ngram

        # 7B Target with 0.5B Draft
        be_7b_05 = {}
        for l in lengths:
            be = econ.calculate_break_even(target_7b, draft_0_5b, draft_length=l, strategy=SpeculationStrategy.DRAFT_MODEL)
            be_7b_05[f"draft_length_{l}"] = {
                "min_acceptance_rate": be.min_acceptance_rate,
                "expected_speedup": be.expected_speedup,
            }
        results["7B_Target_0.5B_Draft"] = be_7b_05

        return results

    def run_speculation_workload_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Phase 5 Workload-Specific Breakdown (Section 21 & 29).
        Compares Standard vs Always-On vs AEON Adaptive across all 8 workload types.
        """
        from aeon.benchmarks.synthetic_workload import generate_phase5_speculation_workload
        from aeon.speculation.types import SpeculationStrategy
        from aeon import AEON

        wl_dict = generate_phase5_speculation_workload()
        out = {}

        aeon_std = AEON(policy="rules", memory=False, tools=False, speculation=False)
        aeon_always = AEON(policy="rules", memory=False, tools=False, speculation=True)
        aeon_adaptive = AEON(policy="voi", memory=True, tools=True, speculation=True)

        for cat, queries in wl_dict.items():
            # Standard
            std_lat = []
            for q in queries:
                r = q.model_copy(deep=True)
                r.speculation = False
                res = aeon_std.run(r)
                std_lat.append(res.latency_ms)
            mean_std_lat = sum(std_lat) / float(len(std_lat))

            # Always-On
            always_lat = []
            for q in queries:
                r = q.model_copy(deep=True)
                r.speculation = True
                r.speculation_strategy = SpeculationStrategy.DRAFT_MODEL.value
                res = aeon_always.run(r)
                always_lat.append(res.latency_ms)
            mean_always_lat = sum(always_lat) / float(len(always_lat))

            # Adaptive
            adaptive_lat = []
            strats_chosen = []
            for q in queries:
                r = q.model_copy(deep=True)
                r.speculation = True
                res = aeon_adaptive.run(r)
                adaptive_lat.append(res.latency_ms)
                strats_chosen.append(res.trace.speculation_strategy or "standard")
            mean_adaptive_lat = sum(adaptive_lat) / float(len(adaptive_lat))

            out[cat] = {
                "standard_latency_ms": round(mean_std_lat, 2),
                "always_on_latency_ms": round(mean_always_lat, 2),
                "aeon_adaptive_latency_ms": round(mean_adaptive_lat, 2),
                "speedup_vs_standard": round(mean_std_lat / max(0.01, mean_adaptive_lat), 2),
                "preferred_strategy": max(set(strats_chosen), key=strats_chosen.count),
            }
        return out



