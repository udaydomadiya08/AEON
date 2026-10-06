"""
AEON Phase 6 — Reasoning Strategies.

Evaluates and selects reasoning strategies based on task classification,
model capabilities, difficulty, and economic constraints. Implements
the strategy transition graph for AEON adaptive reasoning.
"""

import re
from typing import Optional, Tuple, List
from aeon.reasoning.types import (
    ReasoningStrategy,
    ReasoningState,
    ReasoningBudget,
    ReasoningConfig,
    ReasoningStopReason,
)
from aeon.types import TaskType, ReasoningMode


def normalize_reasoning_answer(ans: str) -> str:
    """
    Normalize reasoning answers for robust stability and convergence detection.
    Strips reasoning prefixes, bracketed step counters, whitespace, casing, and terminal punctuation.
    Ensures that identical semantic conclusions from consecutive steps trigger ANSWER_STABLE.
    """
    if not ans:
        return ""
    # Strip prefixes like [Explicit-CoT step 1 (90 tokens)], [Latent-1 (6 compressed states)], etc.
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", ans)
    # Strip common leading labels
    cleaned = re.sub(
        r"^(?:direct\s+answer|reasoned\s+answer|answer)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Normalize whitespace and casing
    cleaned = " ".join(cleaned.lower().strip().split())
    # Strip terminal punctuation
    cleaned = cleaned.rstrip(".;,")
    return cleaned


class ReasoningStrategySelector:
    """
    Selects initial reasoning strategy and manages transitions.
    Maps AEON task types to economically appropriate reasoning depths.
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()

    def select_initial_strategy(
        self,
        task_type: str,
        difficulty: float,
        uncertainty: float,
        model: str,
    ) -> ReasoningStrategy:
        """
        Select the cheapest adequate reasoning strategy for a task.
        AEON principle: never use expensive reasoning when cheap reasoning suffices.
        """
        # Phase 4 Tool integration precedence: deterministic arithmetic / retrieval
        if self.config.enable_tool_assistance and task_type in [
            TaskType.ARITHMETIC.value,
            TaskType.SYMBOLIC_MATH.value,
            TaskType.RETRIEVAL.value,
            "arithmetic",
            "symbolic_math",
            "retrieval",
        ]:
            return ReasoningStrategy.TOOL_ASSISTED

        # No reasoning needed for trivial classification
        if task_type == TaskType.CLASSIFICATION.value and difficulty < 0.35:
            return ReasoningStrategy.NONE

        # Easy tasks: direct answer, no explicit reasoning
        if difficulty < 0.25:
            return ReasoningStrategy.DIRECT

        # Medium tasks: shallow or latent reasoning
        if difficulty < 0.50:
            if task_type in [TaskType.MATH.value, TaskType.CODE.value, TaskType.REASONING.value]:
                if self.config.enable_latent:
                    return ReasoningStrategy.LATENT
                return ReasoningStrategy.SHALLOW
            return ReasoningStrategy.DIRECT

        # Hard tasks: medium or deeper
        if difficulty < 0.70:
            if task_type in [TaskType.MATH.value, TaskType.CODE.value]:
                return ReasoningStrategy.MEDIUM
            if task_type == TaskType.REASONING.value:
                if uncertainty > 0.6:
                    return ReasoningStrategy.MEDIUM
                return ReasoningStrategy.SHALLOW
            return ReasoningStrategy.SHALLOW

        # Very hard tasks: deep/explicit reasoning
        if task_type in [TaskType.MATH.value, TaskType.CODE.value, TaskType.REASONING.value]:
            if not self.config.enable_explicit:
                return ReasoningStrategy.MEDIUM
            if uncertainty > 0.5:
                return ReasoningStrategy.DEEP
            return ReasoningStrategy.EXPLICIT
        return ReasoningStrategy.MEDIUM

    def map_to_reasoning_mode(self, strategy: ReasoningStrategy) -> ReasoningMode:
        """Map Phase 6 reasoning strategy to Phase 1 ReasoningMode for backward compatibility."""
        mapping = {
            ReasoningStrategy.NONE: ReasoningMode.NONE,
            ReasoningStrategy.DIRECT: ReasoningMode.DIRECT,
            ReasoningStrategy.SHALLOW: ReasoningMode.LATENT,
            ReasoningStrategy.MEDIUM: ReasoningMode.LATENT,
            ReasoningStrategy.DEEP: ReasoningMode.EXPLICIT,
            ReasoningStrategy.LATENT: ReasoningMode.LATENT,
            ReasoningStrategy.EXPLICIT: ReasoningMode.EXPLICIT,
            ReasoningStrategy.TOOL_ASSISTED: ReasoningMode.ADAPTIVE,
            ReasoningStrategy.MODEL_ESCALATION: ReasoningMode.EXPLICIT,
            ReasoningStrategy.HYBRID: ReasoningMode.EXPLICIT,
        }
        return mapping.get(strategy, ReasoningMode.DIRECT)

    def should_continue_reasoning(
        self,
        state: ReasoningState,
        budget: ReasoningBudget,
    ) -> Tuple[bool, Optional[ReasoningStopReason]]:
        """
        Check whether reasoning should continue based on convergence criteria.
        Returns (should_continue, stop_reason).
        """
        # Budget exhaustion
        if budget.is_exhausted():
            return False, ReasoningStopReason.BUDGET_EXHAUSTED

        # Max steps
        if state.reasoning_steps >= self.config.max_reasoning_steps:
            return False, ReasoningStopReason.MAX_STEPS_REACHED

        # Confidence target reached
        if state.confidence >= self.config.confidence_target:
            return False, ReasoningStopReason.TARGET_CONFIDENCE_REACHED

        # Quality plateau / stagnation detection
        if len(state.quality_history) >= self.config.stagnation_window:
            recent = state.quality_history[-self.config.stagnation_window:]
            quality_range = max(recent) - min(recent)
            if quality_range < self.config.stagnation_threshold:
                return False, ReasoningStopReason.QUALITY_PLATEAU

        # Confidence stagnation detection
        if len(state.confidence_history) >= self.config.stagnation_window:
            recent = state.confidence_history[-self.config.stagnation_window:]
            conf_range = max(recent) - min(recent)
            if conf_range < self.config.stagnation_threshold:
                state.stagnation_count += 1
                if state.stagnation_count >= 2:
                    return False, ReasoningStopReason.STAGNATION_DETECTED

        # Answer stability: if the last N answers normalize to the same conclusion, stop
        if self.config.enable_answer_stability:
            if len(state.answer_history) >= self.config.answer_stability_window:
                recent_raw = state.answer_history[-self.config.answer_stability_window:]
                recent_norm = [normalize_reasoning_answer(a) for a in recent_raw]
                if len(set(recent_norm)) == 1 and state.confidence >= 0.70:
                    return False, ReasoningStopReason.ANSWER_STABLE

        return True, None

    def estimate_reasoning_budget(
        self,
        strategy: ReasoningStrategy,
        task_type: str,
        difficulty: float,
        uncertainty: float,
        cost_limit: float = 0.005,
        latency_limit: float = 5000.0,
    ) -> ReasoningBudget:
        """Allocate a reasoning-specific budget based on strategy and task."""
        base_tokens = {
            ReasoningStrategy.NONE: 0,
            ReasoningStrategy.DIRECT: 50,
            ReasoningStrategy.SHALLOW: 100,
            ReasoningStrategy.MEDIUM: 300,
            ReasoningStrategy.DEEP: 600,
            ReasoningStrategy.LATENT: 20,
            ReasoningStrategy.EXPLICIT: 400,
            ReasoningStrategy.TOOL_ASSISTED: 150,
            ReasoningStrategy.MODEL_ESCALATION: 200,
            ReasoningStrategy.HYBRID: 500,
        }
        base = base_tokens.get(strategy, 100)

        # Scale by difficulty
        scale = 1.0 + max(0.0, difficulty) * 0.6 + max(0.0, uncertainty) * 0.3

        safe_cost_limit = max(0.0, float(cost_limit))
        safe_latency_limit = max(0.0, float(latency_limit))
        safe_max_tokens = max(0, int(self.config.max_reasoning_tokens))
        safe_max_steps = max(0, int(self.config.max_reasoning_steps))

        base_steps = {
            ReasoningStrategy.NONE: 0,
            ReasoningStrategy.DIRECT: 1,
            ReasoningStrategy.SHALLOW: 2,
            ReasoningStrategy.MEDIUM: 4,
            ReasoningStrategy.DEEP: 8,
            ReasoningStrategy.LATENT: 3,
            ReasoningStrategy.EXPLICIT: 6,
            ReasoningStrategy.TOOL_ASSISTED: 3,
            ReasoningStrategy.MODEL_ESCALATION: 3,
            ReasoningStrategy.HYBRID: 6,
        }
        req_steps = base_steps.get(strategy, 3)

        max_tokens = min(safe_max_tokens, int(base * scale))
        max_steps = min(safe_max_steps, req_steps)

        return ReasoningBudget(
            max_reasoning_tokens=max_tokens,
            max_reasoning_steps=max_steps,
            max_latent_steps=min(self.config.max_latent_steps, max_steps * 2),
            max_latency_ms=safe_latency_limit,
            max_cost_usd=safe_cost_limit,
            max_compute_flops=1e13,
        )

    def get_transition_candidates(self, strategy: ReasoningStrategy) -> List[ReasoningStrategy]:
        """Allowed next strategies in the transition graph with ablation awareness."""
        transitions = {
            ReasoningStrategy.NONE: [ReasoningStrategy.DIRECT, ReasoningStrategy.SHALLOW],
            ReasoningStrategy.DIRECT: [ReasoningStrategy.SHALLOW, ReasoningStrategy.LATENT],
            ReasoningStrategy.SHALLOW: [ReasoningStrategy.MEDIUM, ReasoningStrategy.LATENT],
            ReasoningStrategy.LATENT: [ReasoningStrategy.MEDIUM, ReasoningStrategy.EXPLICIT],
            ReasoningStrategy.MEDIUM: [ReasoningStrategy.DEEP, ReasoningStrategy.EXPLICIT, ReasoningStrategy.TOOL_ASSISTED],
            ReasoningStrategy.EXPLICIT: [ReasoningStrategy.DEEP, ReasoningStrategy.TOOL_ASSISTED, ReasoningStrategy.MODEL_ESCALATION],
            ReasoningStrategy.DEEP: [ReasoningStrategy.MODEL_ESCALATION, ReasoningStrategy.HYBRID],
            ReasoningStrategy.TOOL_ASSISTED: [ReasoningStrategy.HYBRID, ReasoningStrategy.MODEL_ESCALATION],
            ReasoningStrategy.MODEL_ESCALATION: [ReasoningStrategy.HYBRID],
            ReasoningStrategy.HYBRID: [],
        }
        allowed = transitions.get(strategy, [ReasoningStrategy.DEEP])
        filtered = []
        for cand in allowed:
            if cand == ReasoningStrategy.LATENT and not self.config.enable_latent:
                continue
            if cand in [ReasoningStrategy.EXPLICIT, ReasoningStrategy.DEEP] and not self.config.enable_explicit:
                continue
            if cand == ReasoningStrategy.TOOL_ASSISTED and not self.config.enable_tool_assistance:
                continue
            if cand == ReasoningStrategy.MODEL_ESCALATION and not self.config.enable_model_escalation:
                continue
            filtered.append(cand)
        return filtered
