"""
AEON Phase 7 — Depth Controller.

Central orchestration engine for Adaptive Model Depth & Mixture-of-Depths (MoD).
Coordinates early-exit evaluation, token importance estimation, MoD capacity routing,
economic utility optimization, and telemetry production.
"""

import time
import uuid
from typing import Tuple, Dict, Any, Optional, List

from aeon.depth.types import (
    DepthStrategy,
    DepthExitDecision,
    ModelDepthProfile,
    DepthExecutionState,
    TokenComputeState,
    DepthConfig,
)
from aeon.depth.profiles import ModelDepthRegistry, global_depth_registry
from aeon.depth.cost_model import DepthCostModel, global_depth_cost_model
from aeon.depth.token_importance import TokenImportanceEstimator, global_token_importance_estimator
from aeon.depth.capacity_controller import CapacityController, global_capacity_controller
from aeon.depth.mod_router import MoDRouter, global_mod_router
from aeon.depth.early_exit import EarlyExitEvaluator, global_early_exit_evaluator
from aeon.depth.economics import DepthEconomics, global_depth_economics
from aeon.depth.backend import SimulationDepthBackend, global_simulation_depth_backend
from aeon.depth.telemetry import build_depth_telemetry_dict
from aeon.types import UtilityWeights


class DepthController:
    """
    AEON Compute Controller subsystem for model-level and token-level depth optimization.
    """

    def __init__(
        self,
        config: Optional[DepthConfig] = None,
        registry: Optional[ModelDepthRegistry] = None,
        cost_model: Optional[DepthCostModel] = None,
        importance_estimator: Optional[TokenImportanceEstimator] = None,
        capacity_controller: Optional[CapacityController] = None,
        mod_router: Optional[MoDRouter] = None,
        early_exit_evaluator: Optional[EarlyExitEvaluator] = None,
        economics: Optional[DepthEconomics] = None,
        backend: Optional[SimulationDepthBackend] = None,
        weights: Optional[UtilityWeights] = None,
    ):
        self.config = config or DepthConfig()
        self.registry = registry or global_depth_registry
        self.cost_model = cost_model or global_depth_cost_model
        if importance_estimator is not None:
            self.importance_estimator = importance_estimator
        elif not self.config.enable_token_importance:
            from aeon.depth.types import TokenImportanceSignal
            self.importance_estimator = TokenImportanceEstimator(mode=TokenImportanceSignal.UNIFORM)
        else:
            self.importance_estimator = global_token_importance_estimator

        self.capacity_controller = capacity_controller or CapacityController(
            default_capacity=self.config.mod_capacity_ratio
        )
        self.mod_router = mod_router or global_mod_router
        self.early_exit_evaluator = early_exit_evaluator or EarlyExitEvaluator(config=self.config)
        self.economics = economics or DepthEconomics(
            cost_model=self.cost_model, config=self.config, weights=weights
        )
        self.backend = backend or global_simulation_depth_backend
        self.weights = weights or UtilityWeights()

    def execute_depth_inference(
        self,
        query: Any = "",
        context: Optional[Any] = None,
        model_name: Optional[str] = None,
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
        target_confidence: Optional[float] = None,
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        preferred_strategy: Optional[DepthStrategy] = None,
        require_verification: bool = False,
        verification_passed: bool = False,
        **kwargs,
    ) -> Tuple[str, DepthExecutionState, Dict[str, Any]]:
        """
        Execute depth-optimized model inference and return (clean_answer, state, telemetry).
        """
        start_overhead = time.perf_counter()
        request_id = f"depth-{uuid.uuid4().hex[:8]}"

        # Resolve query and request arguments
        if hasattr(query, "query"):
            req = query
            query_str = req.query
            context_str = req.context
            if cost_budget_usd is None:
                cost_budget_usd = req.cost_budget_usd
            if latency_budget_ms is None:
                latency_budget_ms = req.sla_latency_ms
            if not require_verification and req.require_verification:
                require_verification = True
        else:
            query_str = str(query or "")
            context_str = str(context) if isinstance(context, str) else None

        # Resolve model from context if context is a route object
        if context is not None and not isinstance(context, str):
            route_obj = context
            if hasattr(route_obj, "model"):
                model_name = route_obj.model
            elif hasattr(route_obj, "model_name"):
                model_name = route_obj.model_name

        resolved_model = model_name or "aeon-small-1.5b"

        # 1. Profile retrieval
        profile = self.registry.get(resolved_model)
        if not profile:
            from aeon.depth.profiles import get_model_depth_profile
            profile = get_model_depth_profile(resolved_model)

        # 2. Tokenize and estimate importance
        token_states = self.importance_estimator.tokenize_and_estimate(
            text=f"{query_str} {context_str or ''}".strip(),
            difficulty=difficulty,
            uncertainty=uncertainty,
            min_layers=profile.min_safe_depth,
            max_layers=profile.total_layers,
        )
        tokens_count = max(1, len(token_states))

        # 3. Strategy Selection
        if preferred_strategy is not None:
            strategy = preferred_strategy
        elif difficulty <= 0.35 and profile.supported_early_exit and self.config.enable_early_exit:
            strategy = DepthStrategy.ADAPTIVE_EARLY_EXIT
        else:
            candidates = self.economics.evaluate_candidates(
                profile=profile,
                tokens_count=tokens_count,
                difficulty=difficulty,
                uncertainty=uncertainty,
                cost_budget_usd=cost_budget_usd,
                latency_budget_ms=latency_budget_ms,
            )
            viable = [c for c in candidates if c.is_viable]
            strategy = viable[0].strategy if viable else DepthStrategy.FULL_DEPTH

        effective_max_depth = profile.total_layers
        if kwargs.get("max_depth") is not None:
            effective_max_depth = max(1, min(profile.total_layers, int(kwargs["max_depth"])))
        elif hasattr(query, "max_depth") and getattr(query, "max_depth", None) is not None:
            effective_max_depth = max(1, min(profile.total_layers, int(query.max_depth)))

        # 4. Initialize State
        state = DepthExecutionState(
            request_id=request_id,
            model_name=model_name,
            strategy=strategy,
            total_layers=profile.total_layers,
            requested_depth=effective_max_depth,
            min_depth=profile.min_safe_depth,
            max_depth=effective_max_depth,
            tokens_count=tokens_count,
            uncertainty=uncertainty,
            execution_mode="SIMULATED_DEPTH_EXECUTION",
        )

        try:
            # 5. Route Computation based on selected strategy
            if strategy in [DepthStrategy.ADAPTIVE_MOD, DepthStrategy.FIXED_MOD, DepthStrategy.HYBRID_DEPTH] and profile.supported_mod and self.config.enable_mod:
                # Mixture-of-Depths branch
                capacity = (
                    self.capacity_controller.compute_capacity(
                        difficulty=difficulty,
                        uncertainty=uncertainty,
                        target_quality=target_confidence or self.config.target_quality,
                        cost_budget_usd=cost_budget_usd,
                        latency_budget_ms=latency_budget_ms,
                    )
                    if (strategy != DepthStrategy.FIXED_MOD and self.config.enable_adaptive_capacity)
                    else self.config.mod_capacity_ratio
                )

                routed_tokens = self.mod_router.route_token_depths(
                    token_states=token_states,
                    total_layers=profile.total_layers,
                    min_depth=min(profile.min_safe_depth, effective_max_depth),
                    max_depth=effective_max_depth,
                    capacity_ratio=capacity,
                )
                state.token_states = routed_tokens
                state.mod_occurred = True

                res = self.backend.execute_mod(query_str, profile, routed_tokens)
                answer = res["answer"]
                state.estimated_flops = res["flops"]
                state.estimated_latency_ms = res["latency_ms"]
                state.estimated_cost_usd = res["cost_usd"]
                state.avg_token_depth = res["avg_depth"]
                state.actual_depth = int(round(res["avg_depth"]))
                state.deep_tokens = res["deep_tokens"]
                state.shallow_tokens = res["shallow_tokens"]
                state.active_tokens = sum(1 for t in routed_tokens if t.allocated_layers > profile.min_safe_depth)
                state.skipped_tokens = sum(1 for t in routed_tokens if t.allocated_layers == profile.min_safe_depth)
                state.executed_layers = int(round(res["avg_depth"]))
                state.skipped_layers = max(0, profile.total_layers - state.actual_depth)
                state.layer_execution_ratio = res["avg_depth"] / float(profile.total_layers)
                state.exit_decision = DepthExitDecision.EXIT if state.actual_depth < profile.total_layers else DepthExitDecision.CONTINUE
                state.stop_reason = "mod_capacity_completed"
                if state.actual_depth < profile.total_layers:
                    state.early_exit = True
                    state.early_exit_occurred = True
                    state.early_exit_reason = "mod_capacity_completed"

            elif strategy in [DepthStrategy.ADAPTIVE_EARLY_EXIT, DepthStrategy.FIXED_EARLY_EXIT] and profile.supported_early_exit and self.config.enable_early_exit:
                # Early Exit branch
                if strategy == DepthStrategy.FIXED_EARLY_EXIT:
                    target_layer = max(profile.min_safe_depth, profile.total_layers // 2)
                else:
                    # Adaptive early exit: scan exit points
                    target_layer = profile.total_layers
                    for checkpoint in profile.exit_points:
                        if checkpoint < profile.min_safe_depth:
                            continue
                        q_at_layer = profile.quality_curve.get(checkpoint, 0.85)
                        gain_to_full = profile.quality_curve.get(profile.total_layers, 0.95) - q_at_layer
                        cost_layer = self.cost_model.calculate_layer_cost_usd(profile.model_name, profile.total_layers, checkpoint, tokens_count)
                        
                        should_exit, exit_dec, exit_rsn = self.early_exit_evaluator.evaluate_exit(
                            current_layer=checkpoint,
                            profile=profile,
                            current_confidence=q_at_layer - uncertainty * 0.1,
                            current_quality=q_at_layer,
                            marginal_quality_gain=gain_to_full,
                            marginal_cost_usd=cost_layer,
                            require_verification=require_verification,
                            verification_passed=verification_passed,
                            remaining_cost_usd=cost_budget_usd,
                            remaining_latency_ms=latency_budget_ms,
                            target_quality=target_confidence or self.config.target_quality,
                            difficulty=difficulty,
                        )
                        if should_exit:
                            target_layer = checkpoint
                            state.exit_decision = exit_dec
                            state.stop_reason = exit_rsn
                            state.early_exit_occurred = True
                            state.early_exit = True
                            state.early_exit_reason = exit_rsn
                            break

                target_layer = min(effective_max_depth, target_layer)
                res = self.backend.execute_uniform_depth(query_str, profile, target_layer, tokens_count)
                answer = res["answer"]
                state.actual_depth = res["executed_layers"]
                state.executed_layers = res["executed_layers"]
                state.skipped_layers = max(0, profile.total_layers - res["executed_layers"])
                state.layer_execution_ratio = res["executed_layers"] / float(profile.total_layers)
                state.estimated_flops = res["flops"]
                state.estimated_latency_ms = res["latency_ms"]
                state.estimated_cost_usd = res["cost_usd"]
                state.avg_token_depth = float(res["executed_layers"])
                state.active_tokens = tokens_count
                state.deep_tokens = tokens_count if res["executed_layers"] >= profile.total_layers * 0.8 else 0
                state.shallow_tokens = tokens_count if res["executed_layers"] <= profile.min_safe_depth else 0

            else:
                # Full Depth Baseline (clamped to effective_max_depth if constrained)
                exec_layers = min(effective_max_depth, profile.total_layers)
                res = self.backend.execute_uniform_depth(query_str, profile, exec_layers, tokens_count)
                answer = res["answer"]
                state.actual_depth = res["executed_layers"]
                state.executed_layers = res["executed_layers"]
                state.skipped_layers = max(0, profile.total_layers - res["executed_layers"])
                state.layer_execution_ratio = res["executed_layers"] / float(profile.total_layers)
                state.estimated_flops = res["flops"]
                state.estimated_latency_ms = res["latency_ms"]
                state.estimated_cost_usd = res["cost_usd"]
                state.avg_token_depth = float(res["executed_layers"])
                state.active_tokens = tokens_count
                state.deep_tokens = tokens_count
                state.shallow_tokens = 0
                state.exit_decision = DepthExitDecision.CONTINUE
                state.stop_reason = "full_depth_completed"

            # Compute quality and confidence estimates
            curve_q = profile.quality_curve.get(state.actual_depth, 0.85)
            state.quality_estimate = max(0.50, min(1.0, curve_q - difficulty * 0.08))
            if getattr(self.config, "calibrated", True):
                state.confidence = max(0.20, min(0.99, state.quality_estimate - uncertainty * 0.10))
            else:
                state.confidence = min(0.99, state.quality_estimate + 0.12)

            base_q = profile.quality_curve.get(profile.min_safe_depth, 0.65)
            state.marginal_depth_gain = max(0.0, state.quality_estimate - base_q)
            state.depth_voi = (state.marginal_depth_gain / state.estimated_cost_usd) if state.estimated_cost_usd > 0 else 0.0

            # Record action in trace
            state.action_trace.append({
                "action": "EXECUTE_DEPTH",
                "strategy": state.strategy.value,
                "model": model_name,
                "layers_executed": state.executed_layers,
                "total_layers": profile.total_layers,
                "early_exit": state.early_exit_occurred,
                "mod_enabled": state.mod_occurred,
                "avg_token_depth": state.avg_token_depth,
                "cost_usd": state.estimated_cost_usd,
                "latency_ms": state.estimated_latency_ms,
                "flops": state.estimated_flops,
            })

        except Exception as exc:
            # Safe Fallback to full depth direct answer
            state.fallback_occurred = True
            state.exit_decision = DepthExitDecision.FALLBACK_FULL_DEPTH
            state.stop_reason = f"exception_fallback: {str(exc)}"
            answer = f"Depth fallback answer for: {query_str.strip()[:60]}..."
            state.actual_depth = profile.total_layers
            state.executed_layers = profile.total_layers
            state.skipped_layers = 0
            state.layer_execution_ratio = 1.0

        overhead_ms = (time.perf_counter() - start_overhead) * 1000.0
        state.controller_overhead_ms = round(overhead_ms, 3)

        telemetry = build_depth_telemetry_dict(state, profile, overhead_ms)
        return answer, state, telemetry


global_depth_controller = DepthController()
