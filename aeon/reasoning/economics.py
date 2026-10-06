"""
AEON Phase 6 — Reasoning Economics.

Evaluates marginal value, cost, and utility of reasoning strategies.
Computes ΔU = ΔQ_reasoning - cost(ΔReasoning) for each candidate.
Integrates with the existing VOI framework.
"""

import math
from typing import Optional, List
from aeon.reasoning.types import (
    ReasoningStrategy,
    ReasoningCandidate,
    ReasoningState,
    ReasoningBudget,
    ReasoningConfig,
)
from aeon.types import (
    UtilityWeights,
    TaskType,
)
from aeon.models.model_registry import ModelRegistry, global_registry


class ReasoningEconomics:
    """
    Evaluates economic viability of reasoning strategies.

    Implements the AEON economic principle for reasoning:
    Only reason if expected_marginal_quality_gain > marginal_cost_of_reasoning.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        weights: Optional[UtilityWeights] = None,
        config: Optional[ReasoningConfig] = None,
    ):
        self.registry = registry or global_registry
        self.weights = weights or UtilityWeights()
        self.config = config or ReasoningConfig()

    def evaluate_candidates(
        self,
        state: ReasoningState,
        budget: ReasoningBudget,
        task_type: str,
        difficulty: float,
        uncertainty: float,
    ) -> List[ReasoningCandidate]:
        """
        Generate and evaluate all viable reasoning candidates from current state.
        Returns candidates sorted by marginal utility (descending).
        """
        candidates = []

        # Generate candidate strategies based on current state and budget
        potential_strategies = self._generate_candidate_strategies(
            state, budget, task_type, difficulty, uncertainty
        )

        for strategy, model in potential_strategies:
            candidate = self._evaluate_single_candidate(
                strategy=strategy,
                model=model,
                state=state,
                budget=budget,
                task_type=task_type,
                difficulty=difficulty,
                uncertainty=uncertainty,
            )
            candidates.append(candidate)

        # Sort by marginal utility (best first)
        candidates.sort(key=lambda c: c.marginal_utility, reverse=True)
        return candidates

    def select_best_candidate(
        self,
        state: ReasoningState,
        budget: ReasoningBudget,
        task_type: str,
        difficulty: float,
        uncertainty: float,
    ) -> Optional[ReasoningCandidate]:
        """Select the best viable candidate, or None if stopping is optimal."""
        candidates = self.evaluate_candidates(
            state, budget, task_type, difficulty, uncertainty
        )
        for c in candidates:
            if c.is_viable:
                return c
        return None

    def _evaluate_single_candidate(
        self,
        strategy: ReasoningStrategy,
        model: str,
        state: ReasoningState,
        budget: ReasoningBudget,
        task_type: str,
        difficulty: float,
        uncertainty: float,
    ) -> ReasoningCandidate:
        """Evaluate a single reasoning strategy candidate."""
        profile = self.registry.get(model)
        if not profile:
            return ReasoningCandidate(
                strategy=strategy, model=model, is_viable=False,
                reason="model_not_found"
            )

        # Predict expected token cost for this strategy
        est_tokens = self._estimate_tokens(strategy, difficulty, uncertainty)

        # Predict quality gain
        expected_quality_gain = self._predict_quality_gain(
            strategy, model, state, difficulty, uncertainty
        )

        # Compute marginal costs
        marginal_cost_usd = (
            (est_tokens / 1_000_000.0)
            * (profile.cost_per_1m_input_usd * 0.2 + profile.cost_per_1m_output_usd * 0.8)
        )
        marginal_latency_ms = (
            profile.base_latency_ms * 0.3
            + (est_tokens / max(1.0, profile.tokens_per_second)) * 1000.0
        )
        marginal_compute_flops = 2.0 * (profile.param_count_b * 1e9) * est_tokens

        # Check budget feasibility
        if est_tokens > budget.remaining_tokens():
            return ReasoningCandidate(
                strategy=strategy, model=model, is_viable=False,
                expected_quality_gain=expected_quality_gain,
                expected_cost_usd=marginal_cost_usd,
                expected_latency_ms=marginal_latency_ms,
                expected_compute_flops=marginal_compute_flops,
                expected_tokens=est_tokens,
                reason="budget_tokens_exceeded"
            )
        if marginal_cost_usd > budget.remaining_cost():
            return ReasoningCandidate(
                strategy=strategy, model=model, is_viable=False,
                expected_quality_gain=expected_quality_gain,
                expected_cost_usd=marginal_cost_usd,
                expected_latency_ms=marginal_latency_ms,
                expected_compute_flops=marginal_compute_flops,
                expected_tokens=est_tokens,
                reason="budget_cost_exceeded"
            )
        if marginal_latency_ms > budget.remaining_latency():
            return ReasoningCandidate(
                strategy=strategy, model=model, is_viable=False,
                expected_quality_gain=expected_quality_gain,
                expected_cost_usd=marginal_cost_usd,
                expected_latency_ms=marginal_latency_ms,
                expected_compute_flops=marginal_compute_flops,
                expected_tokens=est_tokens,
                reason="budget_latency_exceeded"
            )

        # Diminishing returns factor based on reasoning steps already taken
        diminishing = math.exp(-0.20 * state.reasoning_steps)
        effective_gain = expected_quality_gain * diminishing

        # Marginal energy
        marginal_energy = (marginal_latency_ms / 1000.0) * (
            25.0 if profile.param_count_b < 5 else 120.0
        )

        # Multi-objective utility: ΔU = quality_gain - Σ λ_i * cost_i
        resource_penalty = (
            self.weights.lambda_latency * marginal_latency_ms
            + self.weights.lambda_cost * marginal_cost_usd
            + self.weights.lambda_compute * marginal_compute_flops
            + self.weights.lambda_energy * marginal_energy
        )
        marginal_utility = (self.weights.quality * effective_gain) - resource_penalty

        # VOI ratio: quality gain per dollar
        voi_ratio = (
            (effective_gain / marginal_cost_usd) if marginal_cost_usd > 0 else 0.0
        )

        is_viable = (
            marginal_utility > 0.0
            and voi_ratio >= self.config.reasoning_voi_threshold
            and effective_gain >= self.config.marginal_value_threshold
        )

        return ReasoningCandidate(
            strategy=strategy,
            model=model,
            expected_quality_gain=round(effective_gain, 5),
            expected_cost_usd=round(marginal_cost_usd, 7),
            expected_latency_ms=round(marginal_latency_ms, 2),
            expected_compute_flops=marginal_compute_flops,
            expected_tokens=est_tokens,
            marginal_utility=round(marginal_utility, 5),
            voi_ratio=round(voi_ratio, 2),
            is_viable=is_viable,
            reason="economically_viable" if is_viable else "low_marginal_value",
        )

    def _predict_quality_gain(
        self,
        strategy: ReasoningStrategy,
        model: str,
        state: ReasoningState,
        difficulty: float,
        uncertainty: float,
    ) -> float:
        """
        Predict expected quality gain from applying a reasoning strategy.
        Higher difficulty = more benefit from reasoning.
        Higher uncertainty = more benefit from reasoning.
        """
        profile = self.registry.get(model)
        if not profile:
            return 0.0

        base_model_quality = profile.quality_score
        current_quality = state.quality_estimate

        # Maximum possible gain
        headroom = max(0.0, base_model_quality - current_quality + 0.05)

        # Strategy effectiveness multipliers
        strategy_multipliers = {
            ReasoningStrategy.NONE: 0.0,
            ReasoningStrategy.DIRECT: 0.10,
            ReasoningStrategy.SHALLOW: 0.20,
            ReasoningStrategy.MEDIUM: 0.35,
            ReasoningStrategy.DEEP: 0.55,
            ReasoningStrategy.LATENT: 0.25,
            ReasoningStrategy.EXPLICIT: 0.45,
            ReasoningStrategy.TOOL_ASSISTED: 0.50,
            ReasoningStrategy.MODEL_ESCALATION: 0.60,
            ReasoningStrategy.HYBRID: 0.65,
        }
        multiplier = strategy_multipliers.get(strategy, 0.15)

        # Difficulty scaling: reasoning helps more on hard tasks
        difficulty_bonus = difficulty * 0.15
        uncertainty_bonus = uncertainty * 0.10

        raw_gain = headroom * (multiplier + difficulty_bonus + uncertainty_bonus)

        # Diminishing returns on repeated reasoning (addressed in caller)
        return max(0.0, min(0.35, raw_gain))

    def _estimate_tokens(
        self,
        strategy: ReasoningStrategy,
        difficulty: float,
        uncertainty: float,
    ) -> int:
        """Estimate token budget for a reasoning strategy."""
        base_tokens = {
            ReasoningStrategy.NONE: 0,
            ReasoningStrategy.DIRECT: 30,
            ReasoningStrategy.SHALLOW: 60,
            ReasoningStrategy.MEDIUM: 150,
            ReasoningStrategy.DEEP: 350,
            ReasoningStrategy.LATENT: 10,  # Latent tokens are compressed
            ReasoningStrategy.EXPLICIT: 200,
            ReasoningStrategy.TOOL_ASSISTED: 80,
            ReasoningStrategy.MODEL_ESCALATION: 100,
            ReasoningStrategy.HYBRID: 180,
        }
        base = base_tokens.get(strategy, 80)

        # Scale by difficulty
        scale = 1.0 + (difficulty * 0.5) + (uncertainty * 0.3)
        return max(1, int(base * scale))

    def _generate_candidate_strategies(
        self,
        state: ReasoningState,
        budget: ReasoningBudget,
        task_type: str,
        difficulty: float,
        uncertainty: float,
    ) -> List[tuple]:
        """
        Generate candidate (strategy, model) pairs based on current state.
        Respects the principle: always evaluate cheaper options first.
        """
        candidates = []
        model = state.model_id

        # If no reasoning started, consider starting from cheapest up
        if not state.reasoning_started or state.strategy == ReasoningStrategy.NONE:
            if difficulty < 0.3:
                candidates.append((ReasoningStrategy.DIRECT, model))
            elif difficulty < 0.55:
                candidates.append((ReasoningStrategy.SHALLOW, model))
                candidates.append((ReasoningStrategy.LATENT, model))
            else:
                candidates.append((ReasoningStrategy.SHALLOW, model))
                candidates.append((ReasoningStrategy.MEDIUM, model))
                candidates.append((ReasoningStrategy.LATENT, model))
                if difficulty >= 0.7:
                    candidates.append((ReasoningStrategy.DEEP, model))
            return candidates

        # From existing reasoning, consider escalation transitions
        current = state.strategy

        if current == ReasoningStrategy.DIRECT:
            candidates.append((ReasoningStrategy.SHALLOW, model))
            candidates.append((ReasoningStrategy.LATENT, model))

        elif current == ReasoningStrategy.SHALLOW:
            candidates.append((ReasoningStrategy.MEDIUM, model))
            candidates.append((ReasoningStrategy.LATENT, model))

        elif current in [ReasoningStrategy.MEDIUM, ReasoningStrategy.LATENT]:
            candidates.append((ReasoningStrategy.DEEP, model))
            candidates.append((ReasoningStrategy.EXPLICIT, model))
            if self.config.enable_tool_assistance:
                candidates.append((ReasoningStrategy.TOOL_ASSISTED, model))

        elif current in [ReasoningStrategy.DEEP, ReasoningStrategy.EXPLICIT]:
            if self.config.enable_model_escalation:
                # Try a bigger model
                bigger = self._get_bigger_model(model)
                if bigger:
                    candidates.append((ReasoningStrategy.EXPLICIT, bigger))
                    candidates.append((ReasoningStrategy.HYBRID, bigger))
            if self.config.enable_tool_assistance:
                candidates.append((ReasoningStrategy.TOOL_ASSISTED, model))

        elif current == ReasoningStrategy.TOOL_ASSISTED:
            candidates.append((ReasoningStrategy.HYBRID, model))
            bigger = self._get_bigger_model(model)
            if bigger:
                candidates.append((ReasoningStrategy.EXPLICIT, bigger))

        elif current == ReasoningStrategy.HYBRID:
            bigger = self._get_bigger_model(model)
            if bigger:
                candidates.append((ReasoningStrategy.HYBRID, bigger))

        # Always consider doing nothing more
        if not candidates:
            return [(ReasoningStrategy.NONE, model)]
        return candidates

    def _get_bigger_model(self, current_model: str) -> Optional[str]:
        """Get next larger model in the AEON tier hierarchy."""
        hierarchy = [
            "aeon-decision-jev",
            "aeon-draft-0.5b",
            "aeon-small-1.5b",
            "aeon-medium-7b",
            "aeon-large-70b",
        ]
        try:
            idx = hierarchy.index(current_model)
            if idx < len(hierarchy) - 1:
                return hierarchy[idx + 1]
        except ValueError:
            pass
        return None

    def estimate_step_cost(self, model: str, strategy: ReasoningStrategy) -> float:
        """Estimate monetary cost for a single reasoning step under the given strategy."""
        profile = self.registry.get(model)
        if not profile:
            return 1e-6
        tokens = 6 if strategy == ReasoningStrategy.LATENT else (20 if strategy in [ReasoningStrategy.NONE, ReasoningStrategy.DIRECT] else 90)
        return (tokens / 1_000_000.0) * (profile.cost_per_1m_input_usd * 0.2 + profile.cost_per_1m_output_usd * 0.8)

    def estimate_step_latency(self, model: str, strategy: ReasoningStrategy) -> float:
        """Estimate execution latency (ms) for a single reasoning step."""
        profile = self.registry.get(model)
        if not profile:
            return 50.0
        tokens = 6 if strategy == ReasoningStrategy.LATENT else (20 if strategy in [ReasoningStrategy.NONE, ReasoningStrategy.DIRECT] else 90)
        return profile.base_latency_ms * 0.15 + (tokens / max(1.0, profile.tokens_per_second)) * 1000.0

    def estimate_step_compute(self, model: str, strategy: ReasoningStrategy) -> float:
        """Estimate compute FLOPs for a single reasoning step."""
        profile = self.registry.get(model)
        if not profile:
            return 1e10
        tokens = 6 if strategy == ReasoningStrategy.LATENT else (20 if strategy in [ReasoningStrategy.NONE, ReasoningStrategy.DIRECT] else 90)
        return 2.0 * (profile.param_count_b * 1e9) * tokens

    def estimate_marginal_quality_gain(
        self,
        state: ReasoningState,
        next_strategy: ReasoningStrategy,
        difficulty: float,
        step_number: int = 1,
    ) -> float:
        """Estimate marginal quality improvement from an additional reasoning step."""
        base_gain = 0.15 * difficulty + 0.05
        diminishing = math.exp(-0.25 * max(0, step_number - 1))
        return round(base_gain * diminishing, 4)

    def calculate_reasoning_voi(
        self,
        marginal_quality_gain: float,
        marginal_cost_usd: float,
    ) -> float:
        """Calculate Value-of-Information ratio (quality gain per dollar)."""
        return round(marginal_quality_gain / marginal_cost_usd, 2) if marginal_cost_usd > 0 else 0.0

    def evaluate_all_candidates(
        self,
        state: ReasoningState,
        budget: ReasoningBudget,
        task_type: str = "reasoning",
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
    ) -> List[ReasoningCandidate]:
        """Convenience alias for evaluate_candidates with default task parameters."""
        return self.evaluate_candidates(
            state=state,
            budget=budget,
            task_type=task_type,
            difficulty=difficulty,
            uncertainty=uncertainty,
        )
