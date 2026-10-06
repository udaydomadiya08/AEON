"""
AEON Phase 6 — Reasoning Controller.

Central orchestrator for adaptive reasoning. Integrates with the existing
ComputeController/DynamicStoppingController, VOI framework, memory hierarchy,
and tool controller. Implements dynamic reasoning depth/strategy selection,
confidence-aware stopping, answer stability, and full telemetry.

Conforms to AEON Specification: Reasoning computation is an economic resource
allocated only when expected marginal quality gain exceeds marginal cost.
"""

import math
import time
import uuid
from typing import Optional, Dict, Any, Tuple, List

from aeon.reasoning.types import (
    ReasoningStrategy,
    ReasoningAction,
    ReasoningStopReason,
    ReasoningBudget,
    ReasoningState,
    ReasoningCandidate,
    ConfidenceEstimate,
    ReasoningConfig,
)
from aeon.reasoning.economics import ReasoningEconomics
from aeon.reasoning.strategies import ReasoningStrategySelector
from aeon.types import (
    InferenceRequest,
    TaskType,
    ReasoningMode,
    UtilityWeights,
)
from aeon.models.model_registry import ModelRegistry, global_registry


class ReasoningController:
    """
    Orchestrates adaptive reasoning as an economic computation decision.

    Workflow:
    1. Classify task → estimate difficulty/uncertainty
    2. Select cheapest-adequate reasoning strategy
    3. Allocate reasoning budget
    4. Execute reasoning loop with VOI-based continuation/escalation
    5. Check confidence, answer stability, diminishing returns
    6. Return answer, state, and structured telemetry
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        config: Optional[ReasoningConfig] = None,
        weights: Optional[UtilityWeights] = None,
    ):
        self.registry = registry or global_registry
        self.config = config or ReasoningConfig()
        self.weights = weights or UtilityWeights()
        self.economics = ReasoningEconomics(
            registry=self.registry, weights=self.weights, config=self.config
        )
        self.strategy_selector = ReasoningStrategySelector(config=self.config)

    def execute_reasoning(
        self,
        query: str,
        context: Optional[str] = None,
        task_type: str = "reasoning",
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
        model: str = "aeon-small-1.5b",
        cost_budget_usd: float = 0.005,
        latency_budget_ms: float = 5000.0,
        target_confidence: Optional[float] = None,
        preferred_strategy: Optional[ReasoningStrategy] = None,
    ) -> Tuple[str, ReasoningState, Dict[str, Any]]:
        """
        Execute adaptive reasoning and return (answer, state, telemetry_dict).

        This is the primary Phase 6 entry point. The reasoning controller:
        - Selects the cheapest adequate strategy (or uses preferred_strategy)
        - Allocates a budget
        - Runs an iterative reasoning loop
        - Stops when confidence, budget, or stability criteria are met
        - Reports full telemetry
        """
        request_id = f"reason-{uuid.uuid4().hex[:8]}"

        # 1. Select initial strategy (cheapest-adequate or user-specified)
        if preferred_strategy is not None:
            initial_strategy = preferred_strategy
        else:
            initial_strategy = self.strategy_selector.select_initial_strategy(
                task_type=task_type,
                difficulty=difficulty,
                uncertainty=uncertainty,
                model=model,
            )

        # Override confidence target if specified
        effective_config = self.config
        if target_confidence is not None:
            effective_config = self.config.model_copy()
            effective_config.confidence_target = target_confidence

        # 2. Allocate reasoning budget
        budget = self.strategy_selector.estimate_reasoning_budget(
            strategy=initial_strategy,
            task_type=task_type,
            difficulty=difficulty,
            uncertainty=uncertainty,
            cost_limit=cost_budget_usd,
            latency_limit=latency_budget_ms,
        )

        # 3. Initialize reasoning state
        state = ReasoningState(
            request_id=request_id,
            query=query,
            task_type=task_type,
            difficulty=difficulty,
            strategy=initial_strategy,
            model_id=model,
            uncertainty=uncertainty,
            budget_remaining_tokens=budget.remaining_tokens(),
            budget_remaining_cost=budget.remaining_cost(),
            latency_remaining_ms=budget.remaining_latency(),
            compute_remaining_flops=budget.max_compute_flops,
            cost_remaining_usd=budget.remaining_cost(),
            execution_mode="SIMULATED_LATENT_REASONING",
        )

        # Direct / trivial answer: no explicit reasoning needed
        if initial_strategy in [ReasoningStrategy.NONE, ReasoningStrategy.DIRECT]:
            answer = self._generate_direct_answer(query, task_type, model)
            state.confidence = self._estimate_no_reasoning_confidence(model, difficulty)
            state.quality_estimate = state.confidence
            state.reasoning_stopped = True
            state.is_terminal = True
            state.stop_reason = (
                ReasoningStopReason.NOT_NEEDED
                if initial_strategy == ReasoningStrategy.NONE
                else ReasoningStopReason.TARGET_CONFIDENCE_REACHED
            )
            state.explicit_tokens = 0
            state.reasoning_tokens = 0
            telemetry = self._build_telemetry(state, budget, query)
            return answer, state, telemetry

        # 4. Reasoning loop
        answer = ""
        start_time = time.perf_counter()

        while True:
            state.reasoning_steps += 1
            state.reasoning_started = True

            # Execute one reasoning step (simulated)
            step_answer, step_confidence, step_tokens, step_cost, step_latency, step_compute = (
                self._execute_reasoning_step(
                    query=query,
                    context=context,
                    state=state,
                    budget=budget,
                )
            )

            # Update budget consumption
            budget.tokens_used += step_tokens
            budget.steps_used += 1
            budget.cost_used_usd += step_cost
            budget.latency_used_ms += step_latency
            budget.compute_used_flops += step_compute

            # Update state
            state.reasoning_tokens += step_tokens
            if state.strategy == ReasoningStrategy.LATENT:
                state.latent_steps += 1
            elif state.strategy in [ReasoningStrategy.EXPLICIT, ReasoningStrategy.DEEP, ReasoningStrategy.HYBRID]:
                state.explicit_tokens += step_tokens
            state.confidence = step_confidence
            state.quality_estimate = step_confidence  # In simulation, quality ≈ confidence
            state.answer_history.append(step_answer)
            state.confidence_history.append(step_confidence)
            state.quality_history.append(state.quality_estimate)

            # Compute marginal gains for telemetry
            if len(state.quality_history) >= 2:
                state.marginal_quality_gain = state.quality_history[-1] - state.quality_history[-2]
            state.marginal_cost = step_cost
            state.marginal_latency = step_latency
            state.marginal_compute = step_compute

            # Update remaining budget in state
            state.budget_remaining_tokens = budget.remaining_tokens()
            state.budget_remaining_cost = budget.remaining_cost()
            state.latency_remaining_ms = budget.remaining_latency()
            state.cost_remaining_usd = budget.remaining_cost()

            answer = step_answer

            # Record action
            state.action_history.append({
                "step": state.reasoning_steps,
                "strategy": state.strategy.value,
                "model": state.model_id,
                "tokens": step_tokens,
                "confidence": round(step_confidence, 4),
                "quality": round(state.quality_estimate, 4),
                "cost_usd": round(step_cost, 7),
                "latency_ms": round(step_latency, 2),
                "marginal_gain": round(state.marginal_quality_gain, 5),
            })

            # 5. Check stopping criteria
            should_continue, stop_reason = self.strategy_selector.should_continue_reasoning(
                state, budget
            )
            if not should_continue:
                state.reasoning_stopped = True
                state.is_terminal = True
                state.stop_reason = stop_reason
                break

            # 6. Evaluate VOI for reasoning escalation
            candidate = self.economics.select_best_candidate(
                state=state,
                budget=budget,
                task_type=task_type,
                difficulty=difficulty,
                uncertainty=uncertainty,
            )

            if candidate is None or not candidate.is_viable:
                # No economically viable escalation — stop
                state.reasoning_stopped = True
                state.is_terminal = True
                state.stop_reason = ReasoningStopReason.LOW_MARGINAL_VALUE
                state.voi = 0.0
                break

            # Update state with escalation decision
            state.voi = candidate.voi_ratio
            state.expected_utility = candidate.marginal_utility
            if candidate.strategy != state.strategy or candidate.model != state.model_id:
                state.strategy = candidate.strategy
                state.model_id = candidate.model
                state.action_history.append({
                    "step": state.reasoning_steps,
                    "action": "ESCALATE",
                    "from_strategy": state.strategy.value,
                    "to_strategy": candidate.strategy.value,
                    "from_model": state.model_id,
                    "to_model": candidate.model,
                    "expected_gain": round(candidate.expected_quality_gain, 5),
                    "expected_cost": round(candidate.expected_cost_usd, 7),
                    "voi_ratio": round(candidate.voi_ratio, 2),
                })

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        budget.latency_used_ms = max(budget.latency_used_ms, elapsed_ms)

        # Compute realized utility
        total_cost_penalty = (
            self.weights.lambda_cost * budget.cost_used_usd
            + self.weights.lambda_latency * budget.latency_used_ms
            + self.weights.lambda_compute * budget.compute_used_flops
        )
        state.realized_utility = round(
            self.weights.quality * state.quality_estimate - total_cost_penalty, 5
        )

        # Build telemetry
        telemetry = self._build_telemetry(state, budget, query)
        return answer, state, telemetry

    def _execute_reasoning_step(
        self,
        query: str,
        context: Optional[str],
        state: ReasoningState,
        budget: ReasoningBudget,
    ) -> Tuple[str, float, int, float, float, float]:
        """
        Execute one simulated reasoning step.
        Returns: (answer, confidence, tokens, cost_usd, latency_ms, compute_flops)
        """
        profile = self.registry.get(state.model_id)
        if not profile:
            return ("error", 0.0, 0, 0.0, 0.0, 0.0)

        strategy = state.strategy

        # Token count for this step
        step_tokens = self._step_token_count(strategy, state.difficulty)

        # Simulate answer generation
        answer = self._simulate_reasoning_answer(
            query, strategy, state, step_tokens
        )

        # Calculate confidence from model quality, difficulty, strategy depth, and step
        base_quality = profile.quality_score
        difficulty_penalty = state.difficulty * 0.30
        strategy_boost = self._strategy_confidence_boost(strategy)
        step_improvement = min(0.12, 0.04 * state.reasoning_steps)
        raw_confidence = base_quality - difficulty_penalty + strategy_boost + step_improvement
        uncertainty_penalty = state.uncertainty * 0.15
        confidence = round(max(0.15, min(0.98, raw_confidence - uncertainty_penalty)), 4)

        # Compute resource costs
        cost_usd = (
            (step_tokens / 1_000_000.0)
            * (profile.cost_per_1m_input_usd * 0.2 + profile.cost_per_1m_output_usd * 0.8)
        )
        gen_time_ms = (step_tokens / max(1.0, profile.tokens_per_second)) * 1000.0
        latency_ms = profile.base_latency_ms * 0.15 + gen_time_ms
        compute_flops = 2.0 * (profile.param_count_b * 1e9) * step_tokens

        return answer, confidence, step_tokens, cost_usd, latency_ms, compute_flops

    def _step_token_count(self, strategy: ReasoningStrategy, difficulty: float) -> int:
        """Token count per reasoning step by strategy."""
        base = {
            ReasoningStrategy.NONE: 0,
            ReasoningStrategy.DIRECT: 20,
            ReasoningStrategy.SHALLOW: 35,
            ReasoningStrategy.MEDIUM: 65,
            ReasoningStrategy.DEEP: 120,
            ReasoningStrategy.LATENT: 6,
            ReasoningStrategy.EXPLICIT: 90,
            ReasoningStrategy.TOOL_ASSISTED: 40,
            ReasoningStrategy.MODEL_ESCALATION: 60,
            ReasoningStrategy.HYBRID: 80,
        }
        return max(1, int(base.get(strategy, 30) * (1.0 + difficulty * 0.4)))

    def _strategy_confidence_boost(self, strategy: ReasoningStrategy) -> float:
        """Confidence boost from reasoning strategy."""
        boosts = {
            ReasoningStrategy.NONE: 0.0,
            ReasoningStrategy.DIRECT: 0.02,
            ReasoningStrategy.SHALLOW: 0.05,
            ReasoningStrategy.MEDIUM: 0.10,
            ReasoningStrategy.DEEP: 0.16,
            ReasoningStrategy.LATENT: 0.06,
            ReasoningStrategy.EXPLICIT: 0.14,
            ReasoningStrategy.TOOL_ASSISTED: 0.12,
            ReasoningStrategy.MODEL_ESCALATION: 0.15,
            ReasoningStrategy.HYBRID: 0.18,
        }
        return boosts.get(strategy, 0.03)

    def _simulate_reasoning_answer(
        self,
        query: str,
        strategy: ReasoningStrategy,
        state: ReasoningState,
        tokens: int,
    ) -> str:
        """Generate a simulated reasoning answer."""
        q_lower = query.lower()

        if strategy == ReasoningStrategy.LATENT:
            prefix = f"[Latent-{state.latent_steps + 1} ({tokens} compressed states)]"
        elif strategy in [ReasoningStrategy.EXPLICIT, ReasoningStrategy.DEEP]:
            prefix = f"[Explicit-CoT step {state.reasoning_steps} ({tokens} tokens)]"
        elif strategy == ReasoningStrategy.SHALLOW:
            prefix = f"[Shallow reasoning step {state.reasoning_steps}]"
        elif strategy == ReasoningStrategy.TOOL_ASSISTED:
            prefix = f"[Tool-assisted step {state.reasoning_steps}]"
        elif strategy == ReasoningStrategy.HYBRID:
            prefix = f"[Hybrid reasoning step {state.reasoning_steps}]"
        else:
            prefix = f"[Direct step {state.reasoning_steps}]"

        if "capital" in q_lower and "france" in q_lower:
            return f"{prefix} Paris is the capital of France."
        if "17.3%" in q_lower and "847" in q_lower:
            return f"{prefix} 146.531 (17.3% of 847 = 0.173 * 847 = 146.531)"
        return f"{prefix} Reasoned answer for: {query.strip()[:50]}..."

    def _generate_direct_answer(self, query: str, task_type: str, model: str) -> str:
        """Generate answer without reasoning (NONE strategy)."""
        q_lower = query.lower()
        if "sentiment" in q_lower:
            return "positive" if any(w in q_lower for w in ["good", "great", "love"]) else "negative"
        if "capital" in q_lower and "france" in q_lower:
            return "Paris is the capital of France."
        if "capital" in q_lower and ("italy" in q_lower or "rome" in q_lower):
            return "Rome is the capital of Italy."
        return f"Direct answer: {query.strip()[:60]}..."

    def _estimate_no_reasoning_confidence(self, model: str, difficulty: float) -> float:
        """Confidence when no reasoning is applied."""
        profile = self.registry.get(model)
        base = profile.quality_score if profile else 0.70
        return round(max(0.15, min(0.95, base - difficulty * 0.35)), 4)

    def _build_telemetry(
        self,
        state: ReasoningState,
        budget: ReasoningBudget,
        query: str,
    ) -> Dict[str, Any]:
        """Build structured Phase 6 telemetry."""
        return {
            "reasoning_enabled": True,
            "reasoning_strategy": state.strategy.value,
            "reasoning_model": state.model_id,
            "reasoning_steps": state.reasoning_steps,
            "reasoning_tokens": state.reasoning_tokens,
            "latent_steps": state.latent_steps,
            "explicit_tokens": state.explicit_tokens,
            "tool_calls": state.tool_calls,
            "confidence": round(state.confidence, 4),
            "quality_estimate": round(state.quality_estimate, 4),
            "uncertainty": round(state.uncertainty, 4),
            "marginal_quality_gain": round(state.marginal_quality_gain, 5),
            "marginal_cost_usd": round(state.marginal_cost, 7),
            "marginal_latency_ms": round(state.marginal_latency, 2),
            "marginal_compute_flops": state.marginal_compute,
            "voi": round(state.voi, 2),
            "expected_utility": round(state.expected_utility, 5),
            "realized_utility": round(state.realized_utility, 5),
            "stop_reason": state.stop_reason.value if state.stop_reason else None,
            "budget_tokens_used": budget.tokens_used,
            "budget_tokens_max": budget.max_reasoning_tokens,
            "budget_cost_used": round(budget.cost_used_usd, 7),
            "budget_cost_max": round(budget.max_cost_usd, 7),
            "budget_latency_used": round(budget.latency_used_ms, 2),
            "budget_latency_max": round(budget.max_latency_ms, 2),
            "budget_compute_used": budget.compute_used_flops,
            "budget_compute_max": budget.max_compute_flops,
            "answer_stable": state.stop_reason == ReasoningStopReason.ANSWER_STABLE,
            "stagnation_detected": state.stop_reason in [
                ReasoningStopReason.QUALITY_PLATEAU,
                ReasoningStopReason.STAGNATION_DETECTED,
            ],
            "fallback_occurred": state.fallback_occurred,
            "action_trace": state.action_history,
            "confidence_trace": state.confidence_history,
            "quality_trace": state.quality_history,
            "execution_mode": state.execution_mode,
        }


# Global instance for easy import
global_reasoning_controller = ReasoningController()
