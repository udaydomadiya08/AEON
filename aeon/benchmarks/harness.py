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
