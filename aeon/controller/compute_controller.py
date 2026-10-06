"""
AEON Compute Controller — The Central Architectural Engine.

Receives requests, analyzes complexity, allocates budgets, plans execution routes,
dispatches to inference backends, monitors confidence, and emits structured telemetry.
Conforms to AEON Specification Section 6 and Section 24.
"""

import uuid
import time
from typing import Optional, Dict, Any
from aeon.types import (
    InferenceRequest,
    InferenceResult,
    ExecutionPlan,
    Telemetry,
    TaskType,
    DifficultyLevel,
    RouteChoice,
    ReasoningMode,
    DecisionReason,
)
from aeon.routing.task_classifier import TaskClassifier
from aeon.routing.model_router import ModelRouter
from aeon.routing.cascade_router import CascadeRouter
from aeon.controller.budget_allocator import BudgetAllocator
from aeon.controller.policy_engine import ComputePolicyEngine
from aeon.models.model_registry import ModelRegistry, global_registry
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.inference.simulation_backend import SimulationBackend


class ComputeController:
    """Central intelligent orchestrator for AEON adaptive inference."""

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        registry: Optional[ModelRegistry] = None,
        policy_engine: Optional[ComputePolicyEngine] = None,
        use_cascade: bool = False,
        memory_hierarchy: Optional[Any] = None,
        tool_controller: Optional[Any] = None,
        tools_enabled: bool = False,
        speculation_controller: Optional[Any] = None,
        speculation_enabled: bool = False,
        reasoning_controller: Optional[Any] = None,
        reasoning_enabled: bool = False,
        depth_controller: Optional[Any] = None,
        depth_enabled: bool = False,
        hardware_controller: Optional[Any] = None,
        hardware_enabled: bool = False,
    ):
        self.registry = registry or global_registry
        self.backend = backend or SimulationBackend(self.registry)
        self.classifier = TaskClassifier()
        self.router = ModelRouter()
        self.budget_allocator = BudgetAllocator()
        self.policy_engine = policy_engine or ComputePolicyEngine()
        self.use_cascade = use_cascade
        self.cascade_router = CascadeRouter(self.backend, self.registry)
        self.memory_hierarchy = memory_hierarchy
        self.tool_controller = tool_controller
        self.tools_enabled = tools_enabled
        self.speculation_controller = speculation_controller
        self.speculation_enabled = speculation_enabled
        self.reasoning_controller = reasoning_controller
        self.reasoning_enabled = reasoning_enabled
        self.depth_controller = depth_controller
        self.depth_enabled = depth_enabled
        self.hardware_controller = hardware_controller
        self.hardware_enabled = hardware_enabled

    def plan_execution(self, request: InferenceRequest) -> ExecutionPlan:
        """Analyze request and generate structured execution plan."""
        # 1. Classify task type and estimate intrinsic difficulty
        task_type, difficulty, level, uncertainty = self.classifier.classify(request.query)

        # 2. Allocate component compute budget
        budget = self.budget_allocator.allocate(
            task_type=task_type,
            difficulty=difficulty,
            uncertainty=uncertainty,
            sla_ms=request.sla_latency_ms,
            cost_limit_usd=request.cost_budget_usd,
        )

        # 3. Determine computational route and reasoning mode
        route, reasoning_mode, needs_verification = self.router.route(
            task_type=task_type,
            difficulty=difficulty,
            level=level,
            uncertainty=uncertainty,
            cost_budget_usd=request.cost_budget_usd,
            sla_ms=request.sla_latency_ms,
        )

        selected_model = self.registry.get_model_for_route(route)

        # Check Hardware Optimization (Phase 8 — Hardware, Device, Precision & Quantization)
        hw_active = (request.hardware_aware is True or (request.hardware_aware is None and self.hardware_enabled)) and self.hardware_controller is not None
        hw_device = None
        hw_precision = None
        hw_quant = None
        hw_mode = None
        if hw_active:
            hw_dec = self.hardware_controller.evaluate_and_select(
                query=request.query,
                model_name=selected_model.name,
                difficulty=difficulty,
                uncertainty=uncertainty,
                sla_latency_ms=request.sla_latency_ms,
                cost_budget_usd=request.cost_budget_usd,
                memory_budget_mb=request.memory_budget_mb,
                target_quality=request.target_quality,
                preferred_device=request.device,
                preferred_precision=request.precision,
                preferred_quantization=request.quantization,
            )
            hw_device = hw_dec.device.value
            hw_precision = hw_dec.precision.value
            hw_quant = hw_dec.quantization.value
            hw_mode = hw_dec.execution_mode.value

        return ExecutionPlan(
            task_type=task_type,
            difficulty=difficulty,
            difficulty_level=level,
            uncertainty=uncertainty,
            route=route,
            model=selected_model.name,
            reasoning=reasoning_mode,
            reasoning_budget=int(budget.max_tokens * budget.reasoning),
            max_compute=budget.total_compute_ratio,
            verification=needs_verification,
            cache=True,
            stop_policy="confidence" if route != RouteChoice.LARGE_MODEL else "value_of_compute",
            hardware_device=hw_device,
            precision=hw_precision,
            quantization=hw_quant,
            hardware_execution_mode=hw_mode,
        )

    def run(self, request: InferenceRequest) -> InferenceResult:
        """Execute end-to-end inference request with telemetry."""
        request_id = f"aeon-{uuid.uuid4().hex[:10]}"

        # Memory lookup check
        mem_enabled = request.memory is True or (request.memory is None and self.memory_hierarchy is not None)
        mem_lookup = None
        mem_req = None
        if mem_enabled and self.memory_hierarchy is not None:
            from aeon.memory.types import MemoryRequest, MemoryScope, FreshnessPolicy, ReuseDecision, MemoryLevel
            task_type, diff, lvl, unc = self.classifier.classify(request.query)
            freshness = FreshnessPolicy(request.consistency) if request.consistency else FreshnessPolicy.FRESH
            active_lvls = [MemoryLevel(l) for l in request.memory_levels] if request.memory_levels else [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC]
            mem_req = MemoryRequest(
                query=request.query,
                context=request.context,
                memory_levels=active_lvls,
                task_type=task_type.value,
                scope=MemoryScope.GLOBAL,
                scope_id=request.session_id,
                freshness_policy=freshness,
            )
            mem_lookup = self.memory_hierarchy.lookup(mem_req)
            if mem_lookup.hit and mem_lookup.reuse_decision == ReuseDecision.SAFE_REUSE and mem_lookup.level in [MemoryLevel.L1_KV, MemoryLevel.L4_SEMANTIC, MemoryLevel.L3_INTERMEDIATE]:
                cached_ans = str(mem_lookup.entry.payload) if mem_lookup.entry else ""
                telemetry = Telemetry(
                    request_id=request_id,
                    task_type=task_type.value,
                    route=RouteChoice.SMALL_MODEL.value,
                    model=f"memory-{mem_lookup.level.value}",
                    reasoning_mode=ReasoningMode.NONE.value,
                    input_tokens=mem_lookup.total_tokens,
                    output_tokens=max(5, int(len(cached_ans) / 4.0)),
                    reasoning_tokens=0,
                    latency_ms=round(mem_lookup.lookup_latency_ms, 2),
                    compute_estimate=0.0,
                    memory_mb=10.0,
                    estimated_energy=0.0001,
                    estimated_cost=round(mem_lookup.lookup_cost_usd, 7),
                    cache_hit=True,
                    speculative=False,
                    verification=False,
                    confidence=mem_lookup.entry.confidence if mem_lookup.entry else 0.90,
                    escalated=False,
                    correct=True,
                    latency_type="MEASURED",
                    cost_type="ESTIMATED",
                    compute_type="ESTIMATED",
                    energy_type="ESTIMATED",
                    memory_hit=True,
                    memory_level=mem_lookup.level.value,
                    memory_action="CACHE_HIT",
                    reused_tokens=mem_lookup.total_tokens,
                    compute_saved_flops=mem_lookup.estimated_compute_saved,
                    latency_saved_ms=mem_lookup.estimated_latency_saved_ms,
                    cost_saved_usd=mem_lookup.estimated_cost_saved_usd,
                    energy_saved_joules=round((mem_lookup.estimated_latency_saved_ms / 1000.0) * 25.0, 4),
                    lookup_latency_ms=round(mem_lookup.lookup_latency_ms, 2),
                    memory_similarity=round(mem_lookup.semantic_similarity, 4),
                    freshness_score=1.0,
                    reuse_confidence=mem_lookup.entry.confidence if mem_lookup.entry else 0.90,
                    reuse_decision=mem_lookup.reuse_decision.value,
                    memory_bytes_reused=mem_lookup.entry.size_bytes if mem_lookup.entry else 0,
                )
                return InferenceResult(
                    answer=cached_ans,
                    route=RouteChoice.SMALL_MODEL.value,
                    compute=0.0,
                    cost=round(mem_lookup.lookup_cost_usd, 7),
                    latency_ms=round(mem_lookup.lookup_latency_ms, 2),
                    confidence=telemetry.confidence,
                    trace=telemetry,
                )

        # Tool Execution Check (Phase 4 — Compute-Avoidance Layer)
        tools_enabled = request.tools is True or (request.tools is None and self.tools_enabled)
        if tools_enabled and self.tool_controller is not None:
            task_type, diff, lvl, unc = self.classifier.classify(request.query)
            tool_res = self.tool_controller.evaluate_and_execute(
                query=request.query,
                task_type=task_type,
                difficulty=diff,
                uncertainty=unc,
                budget_constraints=request.budget_constraints,
                require_verification=request.require_verification or False,
            )
            if tool_res is not None:
                tool_ans, top_cand, history, meta = tool_res
                # Memory admission for tool result
                if mem_enabled and self.memory_hierarchy is not None:
                    from aeon.memory.types import MemoryRequest, MemoryScope, FreshnessPolicy, MemoryLevel
                    if mem_req is None:
                        freshness = FreshnessPolicy(request.consistency) if request.consistency else FreshnessPolicy.FRESH
                        mem_req = MemoryRequest(
                            query=request.query,
                            context=request.context,
                            model_id=f"tool-{top_cand.source_name}",
                            task_type=task_type.value,
                            scope=MemoryScope.GLOBAL,
                            scope_id=request.session_id,
                            freshness_policy=freshness,
                        )
                    self.memory_hierarchy.admit_and_store(
                        request=mem_req,
                        payload=tool_ans,
                        level=MemoryLevel.L3_INTERMEDIATE,
                        quality_estimate=top_cand.expected_quality,
                        confidence=top_cand.expected_quality,
                        ttl_seconds=7200.0,
                    )
                    self.memory_hierarchy.admit_and_store(
                        request=mem_req,
                        payload=tool_ans,
                        level=MemoryLevel.L4_SEMANTIC,
                        quality_estimate=top_cand.expected_quality,
                        confidence=top_cand.expected_quality,
                        ttl_seconds=3600.0,
                    )

                telemetry = Telemetry(
                    request_id=request_id,
                    task_type=task_type.value,
                    route=RouteChoice.TOOL.value,
                    model=f"tool-{top_cand.source_name}",
                    reasoning_mode=ReasoningMode.NONE.value,
                    input_tokens=0,
                    output_tokens=max(5, int(len(tool_ans) / 4.0)),
                    reasoning_tokens=0,
                    latency_ms=round(meta["tool_latency_ms"], 2),
                    compute_estimate=meta["tool_compute_flops"],
                    memory_mb=1.0,
                    estimated_energy=round((meta["tool_latency_ms"] / 1000.0) * 10.0, 5),
                    estimated_cost=round(meta["tool_cost_usd"], 7),
                    cache_hit=False,
                    speculative=False,
                    verification=meta["verification_status"] == "verified",
                    confidence=top_cand.expected_quality,
                    escalated=False,
                    correct=True,
                    decision_reason=DecisionReason.TOOL_DETERMINISTIC_SUFFICIENT.value,
                    stopping_reason=DecisionReason.TOOL_DETERMINISTIC_SUFFICIENT.value,
                    latency_type="MEASURED",
                    cost_type="ESTIMATED",
                    compute_type="ESTIMATED",
                    energy_type="ESTIMATED",
                    tool_used=meta["tool_used"],
                    tool_category=meta["tool_category"],
                    tool_cost_usd=round(meta["tool_cost_usd"], 7),
                    tool_latency_ms=round(meta["tool_latency_ms"], 2),
                    tool_compute_flops=meta["tool_compute_flops"],
                    verification_status=meta["verification_status"],
                    unnecessary_model_avoided=True,
                    fallback_occurred=meta["fallback_occurred"],
                    plan_steps=history,
                )

                return InferenceResult(
                    answer=tool_ans,
                    route=RouteChoice.TOOL.value,
                    compute=telemetry.compute_estimate,
                    cost=telemetry.estimated_cost,
                    latency_ms=telemetry.latency_ms,
                    confidence=telemetry.confidence,
                    trace=telemetry,
                    plan=ExecutionPlan(
                        task_type=task_type,
                        difficulty=diff,
                        difficulty_level=lvl,
                        uncertainty=unc,
                        route=RouteChoice.TOOL,
                        model=f"tool-{top_cand.source_name}",
                        reasoning=ReasoningMode.NONE,
                        reasoning_budget=0,
                        tool=top_cand.source_name,
                        max_compute=0.0,
                        verification=meta["verification_status"] == "verified",
                        cache=True,
                        stop_policy="tool_deterministic",
                    ),
                )

        plan = self.plan_execution(request)

        # Reasoning Check (Phase 6 — Adaptive Latent & Token-Efficient Reasoning)
        reasoning_enabled = request.reasoning is True or (request.reasoning is None and self.reasoning_enabled)
        if reasoning_enabled and self.reasoning_controller is not None and plan.route != RouteChoice.DECISION_MODEL:
            task_type, diff, lvl, unc = self.classifier.classify(request.query)
            from aeon.reasoning.types import ReasoningStrategy
            preferred_strat = ReasoningStrategy(request.reasoning_strategy) if request.reasoning_strategy else None
            reason_ans, reason_state, reason_tel = self.reasoning_controller.execute_reasoning(
                query=request.query,
                context=request.context,
                task_type=task_type.value,
                difficulty=diff,
                uncertainty=unc,
                model=plan.model,
                cost_budget_usd=request.cost_budget_usd or 0.005,
                latency_budget_ms=request.sla_latency_ms or 5000.0,
                target_confidence=request.target_confidence,
                preferred_strategy=preferred_strat,
            )

            plan.reasoning_strategy = reason_state.strategy.value
            plan.reasoning_model = reason_state.model_id
            plan.reasoning_budget_tokens = reason_state.reasoning_tokens
            plan.reasoning_depth = reason_state.reasoning_steps

            telemetry = Telemetry(
                request_id=request_id,
                task_type=plan.task_type.value,
                route=plan.route.value,
                model=plan.model,
                reasoning_mode=self._map_reasoning_strategy_to_mode(reason_state.strategy.value),
                input_tokens=max(1, int(len(request.query) / 4.0)),
                output_tokens=max(5, int(len(reason_ans) / 4.0)),
                reasoning_tokens=reason_state.reasoning_tokens,
                latency_ms=round(reason_tel.get("budget_latency_used", 0.0), 2),
                compute_estimate=reason_tel.get("budget_compute_used", 0.0),
                memory_mb=self.registry.get(plan.model).min_memory_mb if self.registry.get(plan.model) else 0.0,
                estimated_energy=round(reason_tel.get("budget_latency_used", 0.0) / 1000.0 * 25.0, 4),
                estimated_cost=round(reason_tel.get("budget_cost_used", 0.0), 7),
                cache_hit=False,
                speculative=False,
                verification=False,
                confidence=round(reason_state.confidence, 4),
                escalated=False,
                correct=True,
                latency_type="MEASURED",
                cost_type="ESTIMATED",
                compute_type="ESTIMATED",
                energy_type="ESTIMATED",
                # Phase 6 Reasoning Telemetry
                reasoning_enabled=True,
                reasoning_strategy=reason_state.strategy.value,
                reasoning_model=reason_state.model_id,
                reasoning_depth_steps=reason_state.reasoning_steps,
                reasoning_latent_steps=reason_state.latent_steps,
                reasoning_explicit_tokens=reason_state.explicit_tokens,
                reasoning_tool_calls=reason_state.tool_calls,
                reasoning_quality_estimate=round(reason_state.quality_estimate, 4),
                reasoning_uncertainty=round(reason_state.uncertainty, 4),
                reasoning_marginal_gain=round(reason_state.marginal_quality_gain, 5),
                reasoning_marginal_cost_usd=round(reason_state.marginal_cost, 7),
                reasoning_marginal_latency_ms=round(reason_state.marginal_latency, 2),
                reasoning_marginal_compute_flops=reason_state.marginal_compute,
                reasoning_voi=round(reason_state.voi, 2),
                reasoning_expected_utility=round(reason_state.expected_utility, 5),
                reasoning_realized_utility=round(reason_state.realized_utility, 5),
                reasoning_stop_reason=reason_state.stop_reason.value if reason_state.stop_reason else None,
                reasoning_budget_tokens_used=reason_state.reasoning_tokens,
                reasoning_budget_tokens_max=reason_tel.get("budget_tokens_max", 0),
                reasoning_budget_cost_used=round(reason_tel.get("budget_cost_used", 0.0), 7),
                reasoning_budget_cost_max=round(reason_tel.get("budget_cost_max", 0.005), 7),
                reasoning_answer_stable=reason_tel.get("answer_stable", False),
                reasoning_stagnation_detected=reason_tel.get("stagnation_detected", False),
                reasoning_confidence_trace=reason_tel.get("confidence_trace", []),
                reasoning_action_trace=reason_tel.get("action_trace", []),
                reasoning_execution_mode=reason_tel.get("execution_mode", "SIMULATED_LATENT_REASONING"),
            )

            if mem_enabled and self.memory_hierarchy is not None and reason_ans:
                from aeon.memory.types import MemoryRequest, MemoryScope, FreshnessPolicy, MemoryLevel
                freshness = FreshnessPolicy(request.consistency) if request.consistency else FreshnessPolicy.FRESH
                mem_store_req = MemoryRequest(
                    query=request.query,
                    context=request.context,
                    model_id=plan.model,
                    task_type=plan.task_type.value,
                    scope=MemoryScope.GLOBAL,
                    scope_id=request.session_id,
                    freshness_policy=freshness,
                )
                self.memory_hierarchy.admit_and_store(
                    request=mem_store_req,
                    payload=reason_ans,
                    level=MemoryLevel.L4_SEMANTIC,
                    quality_estimate=reason_state.quality_estimate,
                    confidence=reason_state.confidence,
                    ttl_seconds=3600.0,
                )

            return InferenceResult(
                answer=reason_ans,
                route=plan.route.value,
                compute=telemetry.compute_estimate,
                cost=telemetry.estimated_cost,
                latency_ms=telemetry.latency_ms,
                confidence=telemetry.confidence,
                trace=telemetry,
                plan=plan,
            )

        # Speculation Check (Phase 5 — Speculative & Token-Efficient Inference)
        spec_enabled = request.speculation is True or (request.speculation is None and self.speculation_enabled)
        if spec_enabled and self.speculation_controller is not None and plan.route != RouteChoice.DECISION_MODEL:
            from aeon.speculation.types import SpeculationStrategy
            preferred_strat = SpeculationStrategy(request.speculation_strategy) if request.speculation_strategy else None
            spec_ans, spec_state, spec_tel = self.speculation_controller.execute_generation(
                target_model=plan.model,
                query=request.query,
                context=request.context,
                task_type=plan.task_type,
                requested_tokens=min(100, max(20, int(len(request.query) * 0.8))),
                preferred_strategy=preferred_strat,
                preferred_draft_model=request.draft_model,
            )
            spec_tel.task_type = plan.task_type.value
            spec_tel.route = plan.route.value
            spec_tel.request_id = request_id

            if mem_enabled and self.memory_hierarchy is not None and spec_ans:
                from aeon.memory.types import MemoryRequest, MemoryScope, FreshnessPolicy, MemoryLevel
                freshness = FreshnessPolicy(request.consistency) if request.consistency else FreshnessPolicy.FRESH
                mem_store_req = MemoryRequest(
                    query=request.query,
                    context=request.context,
                    model_id=plan.model,
                    task_type=plan.task_type.value,
                    scope=MemoryScope.GLOBAL,
                    scope_id=request.session_id,
                    freshness_policy=freshness,
                )
                self.memory_hierarchy.admit_and_store(
                    request=mem_store_req,
                    payload=spec_ans,
                    level=MemoryLevel.L4_SEMANTIC,
                    quality_estimate=0.90,
                    confidence=spec_tel.confidence,
                    ttl_seconds=3600.0,
                )

            plan.speculation_strategy = spec_state.current_strategy.value
            plan.draft_model = spec_state.draft_model
            plan.draft_length = spec_state.draft_length

            return InferenceResult(
                answer=spec_ans,
                route=plan.route.value,
                compute=spec_tel.compute_estimate,
                cost=spec_tel.estimated_cost,
                latency_ms=spec_tel.latency_ms,
                confidence=spec_tel.confidence,
                trace=spec_tel,
                plan=plan,
            )

        # If cascade mode is explicitly active and not on decision model
        depth_tel = {}
        if self.use_cascade and plan.route != RouteChoice.DECISION_MODEL:
            res = self.cascade_router.execute_cascade(
                request=request,
                task_type=plan.task_type,
                difficulty=plan.difficulty,
                level=plan.difficulty_level,
                uncertainty=plan.uncertainty,
            )
            route_str = res["route"]
            model_name = res["final_model"]
            escalated = res.get("escalated", False)
        else:
            # Check Depth Optimization (Phase 7 — Adaptive Model Depth & Mixture-of-Depths)
            depth_active = request.adaptive_depth is True or (request.adaptive_depth is None and self.depth_enabled)
            if depth_active and self.depth_controller is not None:
                from aeon.depth.types import DepthStrategy
                pref_strat = DepthStrategy(request.depth_strategy) if request.depth_strategy else None
                depth_ans, depth_state, depth_tel = self.depth_controller.execute_depth_inference(
                    query=request.query,
                    context=request.context,
                    model_name=plan.model,
                    difficulty=plan.difficulty,
                    uncertainty=plan.uncertainty,
                    target_confidence=request.target_confidence,
                    cost_budget_usd=request.cost_budget_usd,
                    latency_budget_ms=request.sla_latency_ms,
                    preferred_strategy=pref_strat,
                    require_verification=plan.verification or request.require_verification or False,
                    max_depth=getattr(request, "max_depth", None),
                )
                route_str = plan.route.value
                model_name = plan.model
                escalated = False
                is_correct = depth_state.quality_estimate >= (0.75 if plan.difficulty >= 0.60 else 0.60)
                if request.require_verification and not getattr(request, "tools", False):
                    if depth_state.actual_depth < int(round(depth_state.total_layers * 0.70)):
                        is_correct = False
                res = {
                    "answer": depth_ans,
                    "input_tokens": max(1, int(len(request.query) / 4.0)),
                    "output_tokens": max(5, int(len(depth_ans) / 4.0)),
                    "reasoning_tokens": 0,
                    "latency_ms": depth_state.estimated_latency_ms,
                    "compute_estimate": depth_state.estimated_flops,
                    "memory_mb": self.registry.get(plan.model).min_memory_mb if self.registry.get(plan.model) else 0.0,
                    "estimated_energy": round(depth_state.estimated_latency_ms / 1000.0 * 25.0, 4),
                    "estimated_cost": depth_state.estimated_cost_usd,
                    "confidence": depth_state.confidence,
                    "speculative": False,
                    "verification": False,
                    "correct": is_correct,
                }
                plan.depth_strategy = depth_state.strategy.value
                plan.allocated_depth = depth_state.actual_depth
                plan.total_model_layers = depth_state.total_layers
                plan.mod_capacity = depth_tel.get("depth_mod_capacity", 1.0)
            else:
                res = self.backend.generate(request, plan)
                route_str = plan.route.value
                model_name = plan.model
                escalated = False

        latest_answer = res.get("answer", "")
        # Cache admission
        if mem_enabled and self.memory_hierarchy is not None and latest_answer:
            from aeon.memory.types import MemoryRequest, MemoryScope, FreshnessPolicy, MemoryLevel
            if mem_req is None:
                freshness = FreshnessPolicy(request.consistency) if request.consistency else FreshnessPolicy.FRESH
                mem_req = MemoryRequest(
                    query=request.query,
                    context=request.context,
                    model_id=model_name,
                    task_type=plan.task_type.value,
                    scope=MemoryScope.GLOBAL,
                    scope_id=request.session_id,
                    freshness_policy=freshness,
                )
            self.memory_hierarchy.admit_and_store(
                request=mem_req,
                payload=latest_answer,
                level=MemoryLevel.L4_SEMANTIC,
                quality_estimate=0.90,
                confidence=res.get("confidence", 0.90),
                ttl_seconds=3600.0,
            )
            self.memory_hierarchy.admit_and_store(
                request=mem_req,
                payload=latest_answer,
                level=MemoryLevel.L1_KV,
                quality_estimate=0.90,
                confidence=res.get("confidence", 0.90),
                ttl_seconds=1800.0,
            )

        # Check Hardware Optimization (Phase 8 — Hardware, Device, Precision & Quantization)
        hw_active = (request.hardware_aware is True or (request.hardware_aware is None and self.hardware_enabled)) and self.hardware_controller is not None
        hw_dec = None
        if hw_active:
            hw_dec = self.hardware_controller.evaluate_and_select(
                query=request.query,
                model_name=plan.model,
                difficulty=plan.difficulty,
                uncertainty=plan.uncertainty,
                sla_latency_ms=request.sla_latency_ms,
                cost_budget_usd=request.cost_budget_usd,
                memory_budget_mb=request.memory_budget_mb,
                target_quality=request.target_quality,
                preferred_device=request.device,
                preferred_precision=request.precision,
                preferred_quantization=request.quantization,
                depth_layers=plan.allocated_depth if plan.allocated_depth > 0 else None,
                total_layers=plan.total_model_layers if plan.total_model_layers > 0 else None,
            )
            plan.hardware_device = hw_dec.device.value
            plan.precision = hw_dec.precision.value
            plan.quantization = hw_dec.quantization.value
            plan.hardware_execution_mode = hw_dec.execution_mode.value
            if not depth_active:
                res["latency_ms"] = hw_dec.latency_ms
                res["estimated_cost"] = hw_dec.cost_usd
                res["estimated_energy"] = hw_dec.energy_joules
                res["memory_mb"] = hw_dec.memory_mb

        # Build telemetry conforming to Section 20
        telemetry = Telemetry(
            request_id=request_id,
            task_type=plan.task_type.value,
            route=route_str,
            model=model_name,
            reasoning_mode=plan.reasoning.value,
            input_tokens=res.get("input_tokens", 0),
            output_tokens=res.get("output_tokens", 0),
            reasoning_tokens=res.get("reasoning_tokens", 0),
            latency_ms=res.get("latency_ms", 0.0),
            compute_estimate=res.get("compute_estimate", 0.0),
            memory_mb=res.get("memory_mb", 0.0),
            estimated_energy=res.get("estimated_energy", 0.0),
            estimated_cost=res.get("estimated_cost", 0.0),
            cache_hit=False,
            speculative=res.get("speculative", False),
            verification=res.get("verification", False),
            confidence=res.get("confidence", 1.0),
            escalated=escalated,
            correct=res.get("correct"),
            latency_type="MEASURED",
            cost_type="ESTIMATED",
            compute_type="ESTIMATED",
            energy_type="ESTIMATED",
            # Phase 7 Adaptive Model Depth & Mixture-of-Depths (MoD) Telemetry
            depth_enabled=depth_tel.get("depth_enabled", False),
            depth_strategy=depth_tel.get("depth_strategy", None),
            depth_total_layers=depth_tel.get("depth_total_layers", 0),
            depth_requested=depth_tel.get("depth_requested", 0),
            depth_executed=depth_tel.get("depth_executed", 0),
            requested_depth=depth_tel.get("depth_requested", 0),
            actual_depth=depth_tel.get("depth_executed", 0),
            depth_layers_saved=depth_tel.get("depth_skipped_layers", 0),
            depth_skipped_layers=depth_tel.get("depth_skipped_layers", 0),
            depth_layer_ratio=depth_tel.get("depth_layer_ratio", 1.0),
            depth_early_exit=depth_tel.get("depth_early_exit", False),
            early_exit=depth_tel.get("depth_early_exit", False),
            depth_exit_decision=depth_tel.get("depth_exit_decision", None),
            early_exit_reason=depth_tel.get("depth_stop_reason", None),
            depth_stop_reason=depth_tel.get("depth_stop_reason", None),
            depth_mod_enabled=depth_tel.get("depth_mod_enabled", False),
            mod_enabled=depth_tel.get("depth_mod_enabled", False),
            depth_mod_capacity=depth_tel.get("depth_mod_capacity", 1.0),
            mod_capacity=depth_tel.get("depth_mod_capacity", 1.0),
            depth_active_tokens=depth_tel.get("depth_active_tokens", 0),
            mod_active_tokens=depth_tel.get("depth_active_tokens", 0),
            depth_deep_tokens=depth_tel.get("depth_deep_tokens", 0),
            mod_deep_tokens=depth_tel.get("depth_deep_tokens", 0),
            depth_shallow_tokens=depth_tel.get("depth_shallow_tokens", 0),
            mod_shallow_tokens=depth_tel.get("depth_shallow_tokens", 0),
            mod_skipped_tokens=depth_tel.get("depth_skipped_tokens", 0),
            depth_avg_token_depth=depth_tel.get("depth_avg_token_depth", 0.0),
            depth_marginal_gain=depth_tel.get("depth_marginal_gain", 0.0),
            depth_marginal_cost_usd=depth_tel.get("depth_marginal_cost_usd", 0.0),
            depth_voi=depth_tel.get("depth_voi", 0.0),
            depth_estimated_flops=depth_tel.get("depth_estimated_flops", 0.0),
            depth_controller_overhead_ms=depth_tel.get("depth_controller_overhead_ms", 0.0),
            depth_execution_mode=depth_tel.get("depth_execution_mode", "SIMULATED_DEPTH_EXECUTION"),
            # Phase 8 Hardware Telemetry
            hardware_enabled=hw_active,
            hardware_profile_id=self.hardware_controller.profile.profile_id if (hw_active and self.hardware_controller) else None,
            hardware_device=hw_dec.device.value if hw_dec else None,
            hardware_accelerator=self.hardware_controller.profile.accelerator_name if (hw_active and self.hardware_controller) else None,
            precision_requested=request.precision,
            precision_selected=hw_dec.precision.value if hw_dec else "FP32",
            quantization_requested=request.quantization,
            quantization_selected=hw_dec.quantization.value if hw_dec else "none",
            quantization_status=hw_dec.quantization.value if hw_dec else "none",
            memory_total_mb=self.hardware_controller.profile.ram_total_mb if (hw_active and self.hardware_controller) else 0.0,
            memory_available_mb=self.hardware_controller.profile.ram_available_mb if (hw_active and self.hardware_controller) else 0.0,
            memory_estimated_mb=hw_dec.memory_mb if hw_dec else 0.0,
            memory_pressure_level=self.hardware_controller.profile.memory_pressure.value if (hw_active and self.hardware_controller) else "NORMAL",
            thermal_state=self.hardware_controller.profile.thermal_pressure.value if (hw_active and self.hardware_controller) else "NOMINAL",
            energy_joules=hw_dec.energy_joules if hw_dec else 0.0,
            energy_status=hw_dec.execution_mode.value if hw_dec else "ESTIMATED",
            hardware_utility=hw_dec.utility if hw_dec else 0.0,
            hardware_controller_overhead_ms=hw_dec.provenance.get("controller_overhead_ms", 0.0) if hw_dec else 0.0,
            hardware_execution_mode=hw_dec.execution_mode.value if hw_dec else "MEASURED",
            hardware_decision_reason=hw_dec.decision_reason if hw_dec else None,
            hardware_fallbacks=hw_dec.fallbacks_triggered if hw_dec else [],
        )

        return InferenceResult(
            answer=res.get("answer", ""),
            route=route_str,
            compute=telemetry.compute_estimate,
            cost=telemetry.estimated_cost,
            latency_ms=telemetry.latency_ms,
            confidence=telemetry.confidence,
            trace=telemetry,
            plan=plan,
        )

    def _map_reasoning_strategy_to_mode(self, strategy: str) -> str:
        """Map Phase 6 reasoning strategy string to Phase 1 ReasoningMode value."""
        mapping = {
            "none": "none",
            "direct": "direct",
            "shallow": "latent",
            "medium": "latent",
            "deep": "explicit",
            "latent": "latent",
            "explicit": "explicit",
            "tool_assisted": "adaptive",
            "model_escalation": "explicit",
            "hybrid": "explicit",
        }
        return mapping.get(strategy, "direct")
