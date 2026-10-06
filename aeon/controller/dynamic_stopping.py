"""
Dynamic Stopping and Adaptive Compute Controller for AEON.

Executes iterative inference with Value-of-Information (VOI) evaluation,
diminishing-returns modeling, and hard budget constraint enforcement.
Conforms to AEON Specification Sections 8, 13, 14, 15, 16, and 17.
"""

import uuid
import time
from typing import Optional, Dict, Any, List
from aeon.types import (
    InferenceRequest,
    InferenceResult,
    ExecutionPlan,
    Telemetry,
    ComputeState,
    ControllerAction,
    DecisionReason,
    VOIResult,
    RouteChoice,
    ReasoningMode,
    TaskType,
    DifficultyLevel,
    BudgetConstraints,
)
from aeon.controller.voi_estimator import VOIEstimator
from aeon.controller.difficulty_estimator import DifficultyEstimator
from aeon.routing.task_classifier import TaskClassifier
from aeon.models.model_registry import ModelRegistry, global_registry
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.inference.simulation_backend import SimulationBackend
from aeon.decision.calibration import global_calibration_tracker, global_calibration_store


class DynamicStoppingController:
    """
    Iterative computation controller that dynamically decides whether to
    STOP, CONTINUE, or ESCALATE based on marginal quality gains vs. marginal resource costs.
    """

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        registry: Optional[ModelRegistry] = None,
        voi_estimator: Optional[VOIEstimator] = None,
        max_iterations: int = 4,
        memory_hierarchy: Optional[Any] = None,
        tool_controller: Optional[Any] = None,
        tools_enabled: bool = False,
        speculation_controller: Optional[Any] = None,
        speculation_enabled: bool = False,
        reasoning_controller: Optional[Any] = None,
        reasoning_enabled: bool = False,
    ):
        self.registry = registry or global_registry
        self.backend = backend or SimulationBackend(self.registry)
        self.voi_estimator = voi_estimator or VOIEstimator(self.registry)
        self.classifier = TaskClassifier()
        self.max_iterations = max_iterations
        self.memory_hierarchy = memory_hierarchy
        self.tool_controller = tool_controller
        self.tools_enabled = tools_enabled
        self.speculation_controller = speculation_controller
        self.speculation_enabled = speculation_enabled
        self.reasoning_controller = reasoning_controller
        self.reasoning_enabled = reasoning_enabled

    def run(self, request: InferenceRequest) -> InferenceResult:
        request_id = f"aeon-{uuid.uuid4().hex[:10]}"
        last_reason_tel: Optional[Dict[str, Any]] = None
        
        # 1. Initial task classification & difficulty analysis
        task_type, difficulty, level, uncertainty = self.classifier.classify(request.query)

        # Budget constraints resolution
        budget_cfg = request.budget_constraints or BudgetConstraints(
            max_cost_usd=request.cost_budget_usd,
            max_latency_ms=request.sla_latency_ms,
            target_quality=request.target_quality,
        )

        # 2. Memory Lookup (Compute-Avoidance Check — Section 2 & 13)
        mem_enabled = request.memory is True or (request.memory is None and self.memory_hierarchy is not None)
        mem_lookup = None
        mem_req = None
        reused_prefix_tokens = 0

        if mem_enabled and self.memory_hierarchy is not None:
            from aeon.memory.types import (
                MemoryRequest,
                MemoryScope,
                FreshnessPolicy,
                ReuseDecision,
                MemoryLevel,
            )
            freshness = FreshnessPolicy(request.consistency) if request.consistency else FreshnessPolicy.FRESH
            active_lvls = [MemoryLevel(l) for l in request.memory_levels] if request.memory_levels else [MemoryLevel.L1_KV, MemoryLevel.L2_PREFIX, MemoryLevel.L3_INTERMEDIATE, MemoryLevel.L4_SEMANTIC]
            mem_req = MemoryRequest(
                query=request.query,
                context=request.context,
                model_id=None,
                memory_levels=active_lvls,
                task_type=task_type.value,
                scope=MemoryScope.GLOBAL,
                scope_id=request.session_id,
                freshness_policy=freshness,
            )
            mem_lookup = self.memory_hierarchy.lookup(mem_req)

            # Check for immediate SAFE reuse (L1, L3, L4)
            if mem_lookup.hit and mem_lookup.reuse_decision == ReuseDecision.SAFE_REUSE and mem_lookup.level in [MemoryLevel.L1_KV, MemoryLevel.L4_SEMANTIC, MemoryLevel.L3_INTERMEDIATE]:
                cached_ans = str(mem_lookup.entry.payload) if mem_lookup.entry else ""
                cached_q = mem_lookup.entry.quality_estimate if mem_lookup.entry else 0.90
                cached_conf = mem_lookup.entry.confidence if mem_lookup.entry else 0.90

                step_record = {
                    "iteration": 1,
                    "model": f"memory-{mem_lookup.level.value}",
                    "reasoning": ReasoningMode.NONE.value,
                    "confidence": round(cached_conf, 4),
                    "quality_estimate": round(cached_q, 4),
                    "cost_spent_usd": round(mem_lookup.lookup_cost_usd, 7),
                    "latency_spent_ms": round(mem_lookup.lookup_latency_ms, 2),
                    "candidate_action": ControllerAction.CACHE_HIT.value,
                    "marginal_gain": 0.0,
                    "marginal_cost": round(mem_lookup.lookup_cost_usd, 7),
                    "voi": 999.0,
                    "decision": ControllerAction.CACHE_HIT.value,
                    "decision_reason": DecisionReason.CACHE_HIT_SAFE.value,
                }
                trace_steps = [step_record]

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
                    confidence=round(cached_conf, 4),
                    escalated=False,
                    correct=True,
                    initial_quality_estimate=round(cached_q, 4),
                    final_quality_estimate=round(cached_q, 4),
                    marginal_quality_gain=0.0,
                    voi=999.0,
                    marginal_cost=round(mem_lookup.lookup_cost_usd, 7),
                    marginal_latency=round(mem_lookup.lookup_latency_ms, 2),
                    marginal_compute=0.0,
                    decision_reason=DecisionReason.CACHE_HIT_SAFE.value,
                    stopping_reason=DecisionReason.CACHE_HIT_SAFE.value,
                    number_of_controller_iterations=1,
                    number_of_escalations=0,
                    trace_steps=trace_steps,
                    latency_type="MEASURED",
                    cost_type="ESTIMATED",
                    compute_type="ESTIMATED",
                    energy_type="ESTIMATED",
                    memory_hit=True,
                    memory_level=mem_lookup.level.value,
                    memory_action=ControllerAction.CACHE_HIT.value,
                    reused_tokens=mem_lookup.total_tokens,
                    compute_saved_flops=mem_lookup.estimated_compute_saved,
                    latency_saved_ms=mem_lookup.estimated_latency_saved_ms,
                    cost_saved_usd=mem_lookup.estimated_cost_saved_usd,
                    energy_saved_joules=round((mem_lookup.estimated_latency_saved_ms / 1000.0) * 25.0, 4),
                    lookup_latency_ms=round(mem_lookup.lookup_latency_ms, 2),
                    memory_similarity=round(mem_lookup.semantic_similarity, 4),
                    freshness_score=1.0,
                    reuse_confidence=round(cached_conf, 4),
                    reuse_decision=mem_lookup.reuse_decision.value,
                    memory_bytes_reused=mem_lookup.entry.size_bytes if mem_lookup.entry else 0,
                )

                return InferenceResult(
                    answer=cached_ans,
                    route=RouteChoice.SMALL_MODEL.value,
                    compute=0.0,
                    cost=round(mem_lookup.lookup_cost_usd, 7),
                    latency_ms=round(mem_lookup.lookup_latency_ms, 2),
                    confidence=round(cached_conf, 4),
                    trace=telemetry,
                    plan=ExecutionPlan(
                        task_type=task_type,
                        difficulty=difficulty,
                        difficulty_level=level,
                        uncertainty=uncertainty,
                        route=RouteChoice.SMALL_MODEL,
                        model=f"memory-{mem_lookup.level.value}",
                        reasoning=ReasoningMode.NONE,
                        reasoning_budget=0,
                        max_compute=0.0,
                        verification=False,
                        cache=True,
                        stop_policy="cache_hit",
                    ),
                )
            elif mem_lookup.hit and mem_lookup.level == MemoryLevel.L2_PREFIX:
                reused_prefix_tokens = mem_lookup.matched_prefix_tokens

        # 2.5 Tool-Aware Compute Check (Phase 4 — Compute-Avoidance Layer)
        tools_enabled = request.tools is True or (request.tools is None and self.tools_enabled)
        if tools_enabled and self.tool_controller is not None:
            tool_res = self.tool_controller.evaluate_and_execute(
                query=request.query,
                task_type=task_type,
                difficulty=difficulty,
                uncertainty=uncertainty,
                budget_constraints=budget_cfg,
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

                step_record = {
                    "iteration": 1,
                    "model": f"tool-{top_cand.source_name}",
                    "reasoning": ReasoningMode.NONE.value,
                    "confidence": round(top_cand.expected_quality, 4),
                    "quality_estimate": round(top_cand.expected_quality, 4),
                    "cost_spent_usd": round(meta["tool_cost_usd"], 7),
                    "latency_spent_ms": round(meta["tool_latency_ms"], 2),
                    "candidate_action": top_cand.action.value,
                    "marginal_gain": round(top_cand.expected_quality, 4),
                    "marginal_cost": round(meta["tool_cost_usd"], 7),
                    "voi": round(top_cand.voi_ratio, 2),
                    "decision": top_cand.action.value,
                    "decision_reason": DecisionReason.TOOL_DETERMINISTIC_SUFFICIENT.value,
                }
                trace_steps = [step_record]

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
                    confidence=round(top_cand.expected_quality, 4),
                    escalated=False,
                    correct=True,
                    initial_quality_estimate=round(top_cand.expected_quality, 4),
                    final_quality_estimate=round(top_cand.expected_quality, 4),
                    marginal_quality_gain=round(top_cand.expected_quality, 4),
                    voi=round(top_cand.voi_ratio, 2),
                    marginal_cost=round(meta["tool_cost_usd"], 7),
                    marginal_latency=round(meta["tool_latency_ms"], 2),
                    marginal_compute=meta["tool_compute_flops"],
                    decision_reason=DecisionReason.TOOL_DETERMINISTIC_SUFFICIENT.value,
                    stopping_reason=DecisionReason.TOOL_DETERMINISTIC_SUFFICIENT.value,
                    number_of_controller_iterations=1,
                    number_of_escalations=0,
                    trace_steps=trace_steps,
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
                        difficulty=difficulty,
                        difficulty_level=level,
                        uncertainty=uncertainty,
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

        # 3. Select initial economical model tier
        if task_type == TaskType.CLASSIFICATION and level == DifficultyLevel.EASY:
            initial_route = RouteChoice.DECISION_MODEL
            initial_model = "aeon-decision-jev"
            initial_reasoning = ReasoningMode.NONE
        elif level == DifficultyLevel.EASY:
            initial_route = RouteChoice.SMALL_MODEL
            initial_model = "aeon-small-1.5b"
            initial_reasoning = ReasoningMode.DIRECT
        elif level == DifficultyLevel.MEDIUM:
            initial_route = RouteChoice.MEDIUM_MODEL
            initial_model = "aeon-medium-7b"
            initial_reasoning = ReasoningMode.LATENT
        else:  # HARD
            initial_route = RouteChoice.MEDIUM_MODEL
            initial_model = "aeon-medium-7b"
            initial_reasoning = ReasoningMode.LATENT

        # Initialize explicit ComputeState
        state = ComputeState(
            request_id=request_id,
            task_type=task_type,
            difficulty=difficulty,
            difficulty_level=level,
            uncertainty=uncertainty,
            current_model=initial_model,
            current_route=initial_route,
            current_reasoning_mode=initial_reasoning,
            current_quality_estimate=0.0,
            confidence=0.0,
            remaining_cost_budget=budget_cfg.max_cost_usd,
            remaining_latency_ms=budget_cfg.max_latency_ms,
            iteration=0,
            is_terminal=False,
        )

        escalations = 0
        latest_answer = ""
        latest_correct = None
        trace_steps: List[Dict[str, Any]] = []
        initial_q_est: Optional[float] = None
        last_voi_result: Optional[VOIResult] = None
        last_spec_tel: Optional[Telemetry] = None

        # 4. Dynamic Compute Loop
        while not state.is_terminal and state.iteration < self.max_iterations:
            state.iteration += 1

            # Prepare execution plan for current step
            current_profile = self.registry.get(state.current_model)
            plan = ExecutionPlan(
                task_type=state.task_type,
                difficulty=state.difficulty,
                difficulty_level=state.difficulty_level,
                uncertainty=state.uncertainty,
                route=state.current_route,
                model=state.current_model,
                reasoning=state.current_reasoning_mode,
                reasoning_budget=120 if state.current_reasoning_mode == ReasoningMode.EXPLICIT else 0,
                verification=False,
            )

            # Execute model inference step (with speculative decoding if enabled)
            spec_enabled = request.speculation is True or (request.speculation is None and self.speculation_enabled)
            if spec_enabled and self.speculation_controller is not None and plan.route != RouteChoice.DECISION_MODEL:
                from aeon.speculation.types import SpeculationStrategy
                preferred_strat = SpeculationStrategy(request.speculation_strategy) if request.speculation_strategy else None
                spec_ans, spec_st, spec_tel = self.speculation_controller.execute_generation(
                    target_model=state.current_model,
                    query=request.query,
                    context=request.context,
                    task_type=state.task_type,
                    requested_tokens=min(100, max(20, int(len(request.query) * 0.8))),
                    preferred_strategy=preferred_strat,
                    preferred_draft_model=request.draft_model,
                )
                last_spec_tel = spec_tel
                res = {
                    "answer": spec_ans,
                    "input_tokens": 15,
                    "output_tokens": spec_st.generated_tokens,
                    "reasoning_tokens": 0,
                    "latency_ms": spec_tel.latency_ms,
                    "confidence": spec_tel.confidence,
                    "compute_estimate": spec_tel.compute_estimate,
                    "estimated_energy": spec_tel.estimated_energy,
                    "estimated_cost": spec_tel.estimated_cost,
                    "speculative": spec_st.current_strategy != SpeculationStrategy.STANDARD,
                    "verification": True,
                    "correct": True,
                }
            else:
                res = self.backend.generate(request, plan)

            # Reasoning Enhancement (Phase 6) — Apply adaptive reasoning after initial generation if budget permits
            reasoning_enabled = request.reasoning is True or (request.reasoning is None and self.reasoning_enabled)
            rem_cost = state.remaining_cost_budget if state.remaining_cost_budget is not None else 0.005
            rem_lat = state.remaining_latency_ms if state.remaining_latency_ms is not None else 5000.0
            if (
                reasoning_enabled
                and self.reasoning_controller is not None
                and plan.route != RouteChoice.DECISION_MODEL
                and rem_cost > 0.0
                and rem_lat > 0.0
            ):
                from aeon.reasoning.types import ReasoningStrategy
                preferred_strat = ReasoningStrategy(request.reasoning_strategy) if request.reasoning_strategy else None
                reason_ans, reason_state, reason_tel = self.reasoning_controller.execute_reasoning(
                    query=request.query,
                    context=request.context,
                    task_type=state.task_type.value,
                    difficulty=state.difficulty,
                    uncertainty=state.uncertainty,
                    model=state.current_model,
                    cost_budget_usd=rem_cost * 0.5,
                    latency_budget_ms=rem_lat * 0.5,
                    target_confidence=budget_cfg.target_quality,
                    preferred_strategy=preferred_strat,
                )
                last_reason_tel = reason_tel
                # If reasoning produced a confident answer, use it
                if reason_state.confidence > float(res.get("confidence", 0.50)):
                    res["answer"] = reason_ans
                    res["confidence"] = reason_state.confidence
                    res["reasoning_tokens"] = reason_state.reasoning_tokens
                    res["estimated_cost"] = float(res.get("estimated_cost", 0.0)) + reason_tel.get("budget_cost_used", 0.0)
                    res["latency_ms"] = float(res.get("latency_ms", 0.0)) + reason_tel.get("budget_latency_used", 0.0)
                    res["compute_estimate"] = float(res.get("compute_estimate", 0.0)) + reason_tel.get("budget_compute_used", 0.0)

            latest_answer = res.get("answer", "")
            latest_correct = res.get("correct")
            step_confidence = float(res.get("confidence", 0.80))
            step_latency = float(res.get("latency_ms", 0.0))
            step_cost = float(res.get("estimated_cost", 0.0))
            step_compute = float(res.get("compute_estimate", 0.0))
            step_energy = float(res.get("estimated_energy", 0.0))
            step_tokens = int(res.get("input_tokens", 0) + res.get("output_tokens", 0))

            # Update accumulated state
            state.confidence = step_confidence
            state.compute_spent += step_compute
            state.latency_spent_ms += step_latency
            state.monetary_cost_usd += step_cost
            state.energy_spent_joules += step_energy
            state.tokens_generated += step_tokens

            if state.remaining_cost_budget is not None:
                state.remaining_cost_budget = max(0.0, state.remaining_cost_budget - step_cost)
            if state.remaining_latency_ms is not None:
                state.remaining_latency_ms = max(0.0, state.remaining_latency_ms - step_latency)

            # Estimate quality from model profile + task difficulty
            q_est = self.voi_estimator.predict_future_quality(
                state,
                state.current_model,
                state.current_reasoning_mode,
            )
            state.current_quality_estimate = q_est
            if initial_q_est is None:
                initial_q_est = q_est

            # Record empirical calibration data
            global_calibration_tracker.record(step_confidence, res.get("correct", True))
            global_calibration_store.record_run(
                model_name=state.current_model,
                task_type=state.task_type.value,
                difficulty=state.difficulty,
                confidence=step_confidence,
                correct=res.get("correct", True),
                latency_ms=step_latency,
                cost_usd=step_cost,
            )

            # 4. Evaluate VOI for candidate actions from this state
            voi_result = self.voi_estimator.evaluate_candidates(
                state,
                target_quality=budget_cfg.target_quality,
            )
            last_voi_result = voi_result

            # Log step trace for explainability (Section 17 & 28)
            step_record = {
                "iteration": state.iteration,
                "model": state.current_model,
                "reasoning": state.current_reasoning_mode.value,
                "confidence": round(state.confidence, 4),
                "quality_estimate": round(state.current_quality_estimate, 4),
                "cost_spent_usd": round(state.monetary_cost_usd, 7),
                "latency_spent_ms": round(state.latency_spent_ms, 2),
                "candidate_action": voi_result.action.value,
                "marginal_gain": round(voi_result.marginal_quality_gain, 4),
                "marginal_cost": round(voi_result.marginal_cost_usd, 7),
                "voi": round(voi_result.voi_ratio, 2),
                "decision": voi_result.action.value,
                "decision_reason": voi_result.reason.value,
            }
            trace_steps.append(step_record)
            state.history.append(step_record)

            # 5. Handle Controller Decision
            if voi_result.action == ControllerAction.STOP or not voi_result.is_economically_viable:
                state.is_terminal = True
                state.decision_reason = voi_result.reason
                break

            # Handle Escalation or Reasoning Increase
            if voi_result.action in [ControllerAction.ESCALATE_MODEL, ControllerAction.INCREASE_REASONING]:
                escalations += 1
                if voi_result.next_model:
                    state.current_model = voi_result.next_model
                    # Map model name to route choice
                    if "large" in voi_result.next_model:
                        state.current_route = RouteChoice.LARGE_MODEL
                    elif "medium" in voi_result.next_model:
                        state.current_route = RouteChoice.MEDIUM_MODEL
                    else:
                        state.current_route = RouteChoice.SMALL_MODEL

                if voi_result.next_reasoning_mode:
                    state.current_reasoning_mode = voi_result.next_reasoning_mode

        if not state.is_terminal:
            state.decision_reason = DecisionReason.STOP_MAX_ITERATIONS

        # 5. Cache Admission (Section 10 & 13)
        if mem_enabled and self.memory_hierarchy is not None and latest_answer:
            from aeon.memory.types import MemoryRequest, MemoryScope, FreshnessPolicy, MemoryLevel
            if mem_req is None:
                freshness = FreshnessPolicy(request.consistency) if request.consistency else FreshnessPolicy.FRESH
                mem_req = MemoryRequest(
                    query=request.query,
                    context=request.context,
                    model_id=state.current_model,
                    task_type=state.task_type.value,
                    scope=MemoryScope.GLOBAL,
                    scope_id=request.session_id,
                    freshness_policy=freshness,
                )
            # Store in L4 Semantic tier
            self.memory_hierarchy.admit_and_store(
                request=mem_req,
                payload=latest_answer,
                level=MemoryLevel.L4_SEMANTIC,
                quality_estimate=state.current_quality_estimate,
                confidence=state.confidence,
                ttl_seconds=3600.0,
            )
            # Also store exact KV / prompt state in L1 for exact re-queries
            self.memory_hierarchy.admit_and_store(
                request=mem_req,
                payload=latest_answer,
                level=MemoryLevel.L1_KV,
                quality_estimate=state.current_quality_estimate,
                confidence=state.confidence,
                ttl_seconds=1800.0,
            )

        # 6. Build Comprehensive Telemetry
        final_voi = last_voi_result.voi_ratio if last_voi_result else 0.0
        final_marginal_gain = last_voi_result.marginal_quality_gain if last_voi_result else 0.0
        final_marginal_cost = last_voi_result.marginal_cost_usd if last_voi_result else 0.0
        final_marginal_latency = last_voi_result.marginal_latency_ms if last_voi_result else 0.0
        final_marginal_compute = last_voi_result.marginal_compute_flops if last_voi_result else 0.0
        dec_reason = state.decision_reason.value if state.decision_reason else DecisionReason.STOP_LOW_VOI.value

        # Memory telemetry fields
        is_mem_hit = (reused_prefix_tokens > 0)
        mem_lvl = MemoryLevel.L2_PREFIX.value if reused_prefix_tokens > 0 else None
        mem_action = ControllerAction.PARTIAL_CACHE_REUSE.value if reused_prefix_tokens > 0 else None
        comp_saved = mem_lookup.estimated_compute_saved if (mem_lookup and reused_prefix_tokens > 0) else 0.0
        lat_saved = mem_lookup.estimated_latency_saved_ms if (mem_lookup and reused_prefix_tokens > 0) else 0.0
        cost_saved = mem_lookup.estimated_cost_saved_usd if (mem_lookup and reused_prefix_tokens > 0) else 0.0
        lookup_lat = mem_lookup.lookup_latency_ms if mem_lookup else 0.0

        telemetry = Telemetry(
            request_id=request_id,
            task_type=state.task_type.value,
            route=state.current_route.value,
            model=state.current_model,
            reasoning_mode=state.current_reasoning_mode.value,
            input_tokens=max(1, int(len(request.query) / 4.0)),
            output_tokens=max(10, state.tokens_generated - max(1, int(len(request.query) / 4.0))),
            reasoning_tokens=50 if state.current_reasoning_mode == ReasoningMode.EXPLICIT else (8 if state.current_reasoning_mode == ReasoningMode.LATENT else 0),
            latency_ms=round(state.latency_spent_ms, 2),
            compute_estimate=state.compute_spent,
            memory_mb=self.registry.get(state.current_model).min_memory_mb,
            estimated_energy=round(state.energy_spent_joules, 4),
            estimated_cost=round(state.monetary_cost_usd, 7),
            cache_hit=is_mem_hit,
            speculative=last_spec_tel.speculative if last_spec_tel else False,
            verification=last_spec_tel.verification if last_spec_tel else False,
            speculation_strategy=last_spec_tel.speculation_strategy if last_spec_tel else None,
            speculation_action=last_spec_tel.speculation_action if last_spec_tel else None,
            draft_model=last_spec_tel.draft_model if last_spec_tel else None,
            draft_length=last_spec_tel.draft_length if last_spec_tel else 0,
            drafted_tokens=last_spec_tel.drafted_tokens if last_spec_tel else 0,
            accepted_tokens=last_spec_tel.accepted_tokens if last_spec_tel else 0,
            rejected_tokens=last_spec_tel.rejected_tokens if last_spec_tel else 0,
            acceptance_rate=last_spec_tel.acceptance_rate if last_spec_tel else 0.0,
            average_accepted_per_round=last_spec_tel.average_accepted_per_round if last_spec_tel else 0.0,
            draft_latency_ms=last_spec_tel.draft_latency_ms if last_spec_tel else 0.0,
            verification_latency_ms=last_spec_tel.verification_latency_ms if last_spec_tel else 0.0,
            speculation_compute_flops=last_spec_tel.speculation_compute_flops if last_spec_tel else 0.0,
            expected_speedup=last_spec_tel.expected_speedup if last_spec_tel else 1.0,
            realized_speedup=last_spec_tel.realized_speedup if last_spec_tel else 1.0,
            expected_utility=last_spec_tel.expected_utility if last_spec_tel else 0.0,
            realized_utility=last_spec_tel.realized_utility if last_spec_tel else 0.0,
            break_even_acceptance=last_spec_tel.break_even_acceptance if last_spec_tel else 0.0,
            speculation_mode=last_spec_tel.speculation_mode if last_spec_tel else "SIMULATED_SPECULATIVE_DECODING",
            speculation_rounds=last_spec_tel.speculation_rounds if last_spec_tel else 0,
            confidence=round(state.confidence, 4),
            escalated=(escalations > 0),
            correct=latest_correct,
            initial_quality_estimate=round(initial_q_est or 0.0, 4),
            final_quality_estimate=round(state.current_quality_estimate, 4),
            marginal_quality_gain=round(final_marginal_gain, 4),
            voi=round(final_voi, 2),
            marginal_cost=round(final_marginal_cost, 7),
            marginal_latency=round(final_marginal_latency, 2),
            marginal_compute=final_marginal_compute,
            decision_reason=dec_reason,
            stopping_reason=dec_reason,
            budget_before=budget_cfg.max_cost_usd,
            budget_after=state.remaining_cost_budget,
            number_of_controller_iterations=state.iteration,
            number_of_escalations=escalations,
            trace_steps=trace_steps,
            latency_type="MEASURED",
            cost_type="ESTIMATED",
            compute_type="ESTIMATED",
            energy_type="ESTIMATED",
            memory_hit=is_mem_hit,
            memory_level=mem_lvl,
            memory_action=mem_action,
            reused_tokens=reused_prefix_tokens,
            compute_saved_flops=comp_saved,
            latency_saved_ms=lat_saved,
            cost_saved_usd=cost_saved,
            energy_saved_joules=round((lat_saved / 1000.0) * 25.0, 4),
            lookup_latency_ms=round(lookup_lat, 2),
            reuse_decision=mem_lookup.reuse_decision.value if mem_lookup else None,
            # Phase 6 Reasoning Telemetry
            reasoning_enabled=last_reason_tel is not None,
            reasoning_strategy=last_reason_tel.get("reasoning_strategy") if last_reason_tel else None,
            reasoning_model=last_reason_tel.get("reasoning_model") if last_reason_tel else None,
            reasoning_depth_steps=last_reason_tel.get("reasoning_steps", 0) if last_reason_tel else 0,
            reasoning_latent_steps=last_reason_tel.get("latent_steps", 0) if last_reason_tel else 0,
            reasoning_explicit_tokens=last_reason_tel.get("explicit_tokens", 0) if last_reason_tel else 0,
            reasoning_quality_estimate=round(last_reason_tel.get("quality_estimate", 0.0), 4) if last_reason_tel else 0.0,
            reasoning_uncertainty=round(last_reason_tel.get("uncertainty", 0.0), 4) if last_reason_tel else 0.0,
            reasoning_voi=round(last_reason_tel.get("voi", 0.0), 2) if last_reason_tel else 0.0,
            reasoning_stop_reason=last_reason_tel.get("stop_reason") if last_reason_tel else None,
            reasoning_budget_tokens_used=last_reason_tel.get("budget_tokens_used", 0) if last_reason_tel else 0,
            reasoning_budget_tokens_max=last_reason_tel.get("budget_tokens_max", 0) if last_reason_tel else 0,
            reasoning_answer_stable=last_reason_tel.get("answer_stable", False) if last_reason_tel else False,
            reasoning_stagnation_detected=last_reason_tel.get("stagnation_detected", False) if last_reason_tel else False,
            reasoning_confidence_trace=last_reason_tel.get("confidence_trace", []) if last_reason_tel else [],
            reasoning_action_trace=last_reason_tel.get("action_trace", []) if last_reason_tel else [],
            reasoning_execution_mode=last_reason_tel.get("execution_mode", "SIMULATED_LATENT_REASONING") if last_reason_tel else "SIMULATED_LATENT_REASONING",
        )

        final_plan = ExecutionPlan(
            task_type=state.task_type,
            difficulty=state.difficulty,
            difficulty_level=state.difficulty_level,
            uncertainty=state.uncertainty,
            route=state.current_route,
            model=state.current_model,
            reasoning=state.current_reasoning_mode,
            reasoning_budget=120 if state.current_reasoning_mode == ReasoningMode.EXPLICIT else 0,
            max_compute=1.0,
            verification=(escalations > 0),
            stop_policy="value_of_compute",
        )

        return InferenceResult(
            answer=latest_answer,
            route=state.current_route.value,
            compute=state.compute_spent,
            cost=round(state.monetary_cost_usd, 7),
            latency_ms=round(state.latency_spent_ms, 2),
            confidence=round(state.confidence, 4),
            trace=telemetry,
            plan=final_plan,
        )
