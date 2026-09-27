"""
Value-of-Information (VOI) and Diminishing Returns Estimator for AEON.

Evaluates whether an additional unit of computation (model escalation, reasoning steps)
is economically justified by marginal quality gains vs. marginal resource costs.
Conforms to AEON Specification Sections 5, 7, 8, 9, 10, and 15.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
from aeon.types import (
    ComputeState,
    ControllerAction,
    DecisionReason,
    VOIResult,
    UtilityWeights,
    RouteChoice,
    ReasoningMode,
    TaskType,
    DifficultyLevel,
)
from aeon.models.model_registry import ModelRegistry, global_registry


class VOIEstimator:
    """
    Evaluates marginal value of information across candidate computational actions.
    Models diminishing returns on test-time compute and enforces economic rationality.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        weights: Optional[UtilityWeights] = None,
        voi_threshold: float = 25.0,  # Minimum quality gain per dollar to justify continuation
        high_confidence_stop_threshold: float = 0.90,
    ):
        self.registry = registry or global_registry
        self.weights = weights or UtilityWeights()
        self.voi_threshold = voi_threshold
        self.high_confidence_stop_threshold = high_confidence_stop_threshold

    def evaluate_candidates(
        self,
        state: ComputeState,
        target_quality: Optional[float] = None,
    ) -> VOIResult:
        """
        Evaluate candidate actions from the current compute state and select
        the optimal action according to marginal net utility and VOI.
        """
        # 1. Check if current confidence is already very high and task is solved
        if state.confidence >= self.high_confidence_stop_threshold:
            return self._build_stop_result(
                state,
                reason=DecisionReason.STOP_HIGH_CONFIDENCE,
                marginal_utility=0.0,
            )

        # 2. Check if target quality is already achieved
        if target_quality is not None and state.current_quality_estimate >= target_quality:
            return self._build_stop_result(
                state,
                reason=DecisionReason.STOP_TARGET_QUALITY_REACHED,
                marginal_utility=0.0,
            )

        # 3. Check budget exhaustion
        if (state.remaining_cost_budget is not None and state.remaining_cost_budget <= 0.000001) or \
           (state.remaining_latency_ms is not None and state.remaining_latency_ms <= 10.0):
            return self._build_stop_result(
                state,
                reason=DecisionReason.STOP_BUDGET_EXHAUSTED,
                marginal_utility=0.0,
            )

        # 4. Generate candidate actions
        candidates = self._generate_candidate_actions(state)
        if not candidates:
            return self._build_stop_result(
                state,
                reason=DecisionReason.STOP_LOW_VOI,
                marginal_utility=0.0,
            )

        # 5. Evaluate VOI for each candidate
        best_candidate: Optional[VOIResult] = None
        best_marginal_utility = -float("inf")

        for action, next_model, next_reasoning in candidates:
            voi_res = self.estimate_action(state, action, next_model, next_reasoning, target_quality=target_quality)
            
            # Check budget constraints on candidate
            if state.remaining_cost_budget is not None and voi_res.marginal_cost_usd > state.remaining_cost_budget:
                continue
            if state.remaining_latency_ms is not None and voi_res.marginal_latency_ms > state.remaining_latency_ms:
                continue

            if voi_res.marginal_utility > best_marginal_utility:
                best_marginal_utility = voi_res.marginal_utility
                best_candidate = voi_res

        # 6. If no viable candidate has positive marginal utility or VOI >= threshold, stop
        if best_candidate is None or not best_candidate.is_economically_viable:
            return self._build_stop_result(
                state,
                reason=DecisionReason.STOP_LOW_VOI,
                marginal_utility=0.0,
            )

        return best_candidate

    def estimate_action(
        self,
        state: ComputeState,
        action: ControllerAction,
        next_model: str,
        next_reasoning: ReasoningMode,
        target_quality: Optional[float] = None,
    ) -> VOIResult:
        """Calculate exact marginal gains, costs, and VOI for a candidate action."""
        current_q = state.current_quality_estimate
        future_q = self.predict_future_quality(state, next_model, next_reasoning)
        
        # Diminishing returns: marginal gain shrinks as quality approaches 1.0 or reasoning steps increase
        raw_gain = max(0.0, future_q - current_q)
        diminishing_factor = math.exp(-0.25 * state.iteration)
        marginal_quality_gain = round(raw_gain * diminishing_factor, 4)

        # Marginal resource costs
        profile = self.registry.get(next_model)
        est_tokens = 60 if next_reasoning == ReasoningMode.DIRECT else (180 if next_reasoning == ReasoningMode.EXPLICIT else 90)
        
        marginal_cost_usd = (est_tokens / 1_000_000.0) * (profile.cost_per_1m_input_usd * 0.3 + profile.cost_per_1m_output_usd * 0.7)
        marginal_latency_ms = profile.base_latency_ms + (est_tokens / profile.tokens_per_second) * 1000.0
        marginal_compute_flops = 2.0 * (profile.param_count_b * 1e9) * est_tokens
        marginal_energy_joules = (marginal_latency_ms / 1000.0) * (25.0 if profile.param_count_b < 5 else 120.0)

        # If user explicitly requested target_quality that has not yet been achieved, adjust incentives
        quality_weight = self.weights.quality
        effective_threshold = self.voi_threshold
        if target_quality is not None and current_q < target_quality:
            quality_weight *= 3.0
            effective_threshold = min(5.0, effective_threshold)

        # Multi-objective utility: ΔU = Quality_gain - Σ λ_i * Cost_i
        resource_penalty = (
            self.weights.lambda_latency * marginal_latency_ms
            + self.weights.lambda_cost * marginal_cost_usd
            + self.weights.lambda_compute * marginal_compute_flops
            + self.weights.lambda_energy * marginal_energy_joules
        )
        marginal_utility = round((quality_weight * marginal_quality_gain) - resource_penalty, 5)

        # VOI ratio = Marginal Quality Gain / Marginal Cost USD
        voi_ratio = (marginal_quality_gain / marginal_cost_usd) if marginal_cost_usd > 0 else 0.0

        is_economically_viable = (voi_ratio >= effective_threshold) and (marginal_utility > 0.0)

        reason = (
            DecisionReason.ESCALATE_HIGH_EXPECTED_GAIN
            if action == ControllerAction.ESCALATE_MODEL
            else DecisionReason.CONTINUE_HIGH_VOI
        )

        return VOIResult(
            action=action,
            expected_current_quality=round(current_q, 4),
            expected_future_quality=round(current_q + marginal_quality_gain, 4),
            marginal_quality_gain=marginal_quality_gain,
            marginal_cost_usd=round(marginal_cost_usd, 7),
            marginal_latency_ms=round(marginal_latency_ms, 2),
            marginal_compute_flops=marginal_compute_flops,
            marginal_utility=marginal_utility,
            voi_ratio=round(voi_ratio, 2),
            voi_threshold=self.voi_threshold,
            is_economically_viable=is_economically_viable,
            reason=reason,
            next_model=next_model,
            next_reasoning_mode=next_reasoning,
        )

    def estimate_memory_action(
        self,
        state: ComputeState,
        lookup_result: Any,
    ) -> VOIResult:
        """
        Evaluate marginal VOI for reusing cached computation versus recomputing.
        Conforms to AEON Specification Section 11 & Section 30.
        """
        from aeon.memory.types import MemoryLevel, ReuseDecision

        if not lookup_result.hit or lookup_result.reuse_decision == ReuseDecision.REJECT_REUSE:
            return self._build_stop_result(
                state,
                reason=DecisionReason.CACHE_REJECTED_RISK,
                marginal_utility=-0.01,
            )

        cached_q = lookup_result.entry.quality_estimate if lookup_result.entry else 0.90
        current_q = state.current_quality_estimate
        marginal_quality_gain = max(0.0, cached_q - current_q)

        marginal_cost_usd = lookup_result.lookup_cost_usd
        marginal_latency_ms = lookup_result.lookup_latency_ms
        marginal_compute_flops = 0.0

        resource_penalty = (
            self.weights.lambda_latency * marginal_latency_ms
            + self.weights.lambda_cost * marginal_cost_usd
        )
        marginal_utility = round((self.weights.quality * marginal_quality_gain) - resource_penalty + 0.15, 5)
        voi_ratio = (marginal_quality_gain / marginal_cost_usd) if marginal_cost_usd > 0 else 999.0

        is_safe = (lookup_result.reuse_decision == ReuseDecision.SAFE_REUSE)
        action = ControllerAction.CACHE_HIT if is_safe else ControllerAction.VERIFY_CACHE
        if lookup_result.level == MemoryLevel.L2_PREFIX:
            action = ControllerAction.PARTIAL_CACHE_REUSE
            reason = DecisionReason.PARTIAL_PREFIX_REUSE
        else:
            reason = DecisionReason.CACHE_HIT_SAFE if is_safe else DecisionReason.VERIFY_HIGH_UNCERTAINTY

        return VOIResult(
            action=action,
            expected_current_quality=round(current_q, 4),
            expected_future_quality=round(cached_q, 4),
            marginal_quality_gain=marginal_quality_gain,
            marginal_cost_usd=round(marginal_cost_usd, 7),
            marginal_latency_ms=round(marginal_latency_ms, 2),
            marginal_compute_flops=marginal_compute_flops,
            marginal_utility=marginal_utility,
            voi_ratio=round(voi_ratio, 2),
            voi_threshold=self.voi_threshold,
            is_economically_viable=True,
            reason=reason,
            next_model=None,
            next_reasoning_mode=None,
        )

    def estimate_tool_action(
        self,
        state: ComputeState,
        candidate: Any,
    ) -> VOIResult:
        """
        Evaluate marginal VOI for selecting a specialized computational tool over a model.
        Conforms to AEON Specification Section 6, 7, 26.
        """
        current_q = state.current_quality_estimate
        future_q = candidate.expected_quality
        marginal_quality_gain = max(0.0, future_q - current_q)

        marginal_cost = candidate.expected_cost_usd + candidate.verification_cost_usd
        marginal_latency = candidate.expected_latency_ms
        marginal_compute = candidate.expected_compute_flops

        voi_ratio = (marginal_quality_gain / marginal_cost) if marginal_cost > 0 else 9999.0

        is_deterministic = candidate.tool_capability.deterministic if candidate.tool_capability else True
        reason = (
            DecisionReason.TOOL_DETERMINISTIC_SUFFICIENT
            if is_deterministic
            else DecisionReason.TOOL_MORE_EFFICIENT
        )

        return VOIResult(
            action=candidate.action,
            expected_current_quality=round(current_q, 4),
            expected_future_quality=round(future_q, 4),
            marginal_quality_gain=round(marginal_quality_gain, 4),
            marginal_cost_usd=round(marginal_cost, 7),
            marginal_latency_ms=round(marginal_latency, 2),
            marginal_compute_flops=marginal_compute,
            marginal_utility=round(candidate.net_utility, 5),
            voi_ratio=round(voi_ratio, 2),
            voi_threshold=self.voi_threshold,
            is_economically_viable=candidate.net_utility > 0.0,
            reason=reason,
            next_model=None,
            next_reasoning_mode=None,
        )

    def predict_future_quality(
        self,
        state: ComputeState,
        target_model: str,
        target_reasoning: ReasoningMode,
    ) -> float:
        """Predict expected quality for a model and reasoning mode given task difficulty."""
        profile = self.registry.get(target_model)
        if not profile:
            return 0.50

        base_quality = profile.quality_score
        
        # Difficulty penalty: higher difficulty lowers expected accuracy on smaller models
        difficulty_sensitivity = 0.55 if profile.param_count_b < 2.0 else (0.35 if profile.param_count_b < 10.0 else 0.15)
        adjusted_q = base_quality - (state.difficulty * difficulty_sensitivity)

        # Reasoning mode boost
        if target_reasoning == ReasoningMode.EXPLICIT:
            reasoning_boost = 0.12 * state.difficulty
        elif target_reasoning == ReasoningMode.LATENT:
            reasoning_boost = 0.08 * state.difficulty
        else:
            reasoning_boost = 0.0

        # Uncertainty dampener
        uncertainty_penalty = 0.15 * state.uncertainty

        return max(0.20, min(0.99, adjusted_q + reasoning_boost - uncertainty_penalty))

    def _generate_candidate_actions(
        self,
        state: ComputeState,
    ) -> List[Tuple[ControllerAction, str, ReasoningMode]]:
        """Identify allowable computational transitions given current state."""
        candidates = []

        curr_model = state.current_model
        # Transition 1: From Decision Model -> Small Model
        if curr_model == "aeon-decision-jev":
            candidates.append((ControllerAction.ESCALATE_MODEL, "aeon-small-1.5b", ReasoningMode.DIRECT))
            candidates.append((ControllerAction.ESCALATE_MODEL, "aeon-small-1.5b", ReasoningMode.LATENT))
        
        # Transition 2: From Small Model -> Medium Model or Latent -> Explicit
        elif curr_model == "aeon-small-1.5b":
            if state.current_reasoning_mode == ReasoningMode.DIRECT:
                candidates.append((ControllerAction.INCREASE_REASONING, "aeon-small-1.5b", ReasoningMode.LATENT))
            candidates.append((ControllerAction.ESCALATE_MODEL, "aeon-medium-7b", ReasoningMode.LATENT))
            candidates.append((ControllerAction.ESCALATE_MODEL, "aeon-medium-7b", ReasoningMode.EXPLICIT))

        # Transition 3: From Medium Model -> Large Model or Latent -> Explicit
        elif curr_model == "aeon-medium-7b":
            if state.current_reasoning_mode != ReasoningMode.EXPLICIT:
                candidates.append((ControllerAction.INCREASE_REASONING, "aeon-medium-7b", ReasoningMode.EXPLICIT))
            candidates.append((ControllerAction.ESCALATE_MODEL, "aeon-large-70b", ReasoningMode.EXPLICIT))

        # Transition 4: From Large Model with Latent -> Explicit
        elif curr_model == "aeon-large-70b" and state.current_reasoning_mode != ReasoningMode.EXPLICIT:
            candidates.append((ControllerAction.INCREASE_REASONING, "aeon-large-70b", ReasoningMode.EXPLICIT))

        return candidates

    def _build_stop_result(
        self,
        state: ComputeState,
        reason: DecisionReason,
        marginal_utility: float = 0.0,
    ) -> VOIResult:
        return VOIResult(
            action=ControllerAction.STOP,
            expected_current_quality=round(state.current_quality_estimate, 4),
            expected_future_quality=round(state.current_quality_estimate, 4),
            marginal_quality_gain=0.0,
            marginal_cost_usd=0.0,
            marginal_latency_ms=0.0,
            marginal_compute_flops=0.0,
            marginal_utility=marginal_utility,
            voi_ratio=0.0,
            voi_threshold=self.voi_threshold,
            is_economically_viable=False,
            reason=reason,
            next_model=None,
            next_reasoning_mode=None,
        )
