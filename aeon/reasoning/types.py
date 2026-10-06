"""
AEON Phase 6 — Adaptive Latent & Token-Efficient Reasoning.

Type definitions for reasoning strategies, states, actions, budgets,
confidence estimation, and configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReasoningStrategy(str, Enum):
    """Reasoning depth/mode strategy."""
    NONE = "none"
    DIRECT = "direct"
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"
    LATENT = "latent"
    EXPLICIT = "explicit"
    TOOL_ASSISTED = "tool_assisted"
    MODEL_ESCALATION = "model_escalation"
    HYBRID = "hybrid"


class ReasoningAction(str, Enum):
    """Reasoning controller actions."""
    START_REASONING = "START_REASONING"
    CONTINUE_REASONING = "CONTINUE_REASONING"
    INCREASE_REASONING = "INCREASE_REASONING"
    REDUCE_REASONING = "REDUCE_REASONING"
    SWITCH_REASONING_MODE = "SWITCH_REASONING_MODE"
    SWITCH_MODEL = "SWITCH_MODEL"
    INVOKE_TOOL = "INVOKE_TOOL"
    CONVERT_TO_LATENT = "CONVERT_TO_LATENT"
    CONVERT_TO_EXPLICIT = "CONVERT_TO_EXPLICIT"
    VERIFY = "VERIFY"
    STOP_REASONING = "STOP_REASONING"
    ABORT = "ABORT"


class ReasoningStopReason(str, Enum):
    """Why reasoning was stopped."""
    TARGET_CONFIDENCE_REACHED = "TARGET_CONFIDENCE_REACHED"
    LOW_MARGINAL_VALUE = "LOW_MARGINAL_VALUE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    LATENCY_LIMIT = "LATENCY_LIMIT"
    COST_LIMIT = "COST_LIMIT"
    COMPUTE_LIMIT = "COMPUTE_LIMIT"
    VERIFICATION_SUFFICIENT = "VERIFICATION_SUFFICIENT"
    ANSWER_STABLE = "ANSWER_STABLE"
    QUALITY_PLATEAU = "QUALITY_PLATEAU"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    STAGNATION_DETECTED = "STAGNATION_DETECTED"
    FAILURE = "FAILURE"
    NOT_NEEDED = "NOT_NEEDED"


class ReasoningBudget(BaseModel):
    """Dynamic reasoning budget allocation."""
    max_reasoning_tokens: int = 1000
    max_reasoning_steps: int = 8
    max_latent_steps: int = 16
    max_latency_ms: float = 5000.0
    max_cost_usd: float = 0.005
    max_compute_flops: float = 1e13
    max_memory_mb: float = 16000.0
    max_energy_joules: float = 50.0

    # Consumed tracking
    tokens_used: int = 0
    steps_used: int = 0
    latent_steps_used: int = 0
    latency_used_ms: float = 0.0
    cost_used_usd: float = 0.0
    compute_used_flops: float = 0.0
    memory_used_mb: float = 0.0
    energy_used_joules: float = 0.0

    def remaining_tokens(self) -> int:
        return max(0, self.max_reasoning_tokens - self.tokens_used)

    def remaining_steps(self) -> int:
        return max(0, self.max_reasoning_steps - self.steps_used)

    def remaining_cost(self) -> float:
        return max(0.0, self.max_cost_usd - self.cost_used_usd)

    def remaining_latency(self) -> float:
        return max(0.0, self.max_latency_ms - self.latency_used_ms)

    def is_exhausted(self) -> bool:
        return (
            self.tokens_used >= self.max_reasoning_tokens
            or self.steps_used >= self.max_reasoning_steps
            or self.cost_used_usd >= self.max_cost_usd
            or self.latency_used_ms >= self.max_latency_ms
        )


class ReasoningState(BaseModel):
    """Comprehensive state tracking for a reasoning session."""
    request_id: str = ""
    query: str = ""
    task_type: str = ""
    difficulty: float = 0.0
    strategy: ReasoningStrategy = ReasoningStrategy.NONE
    model_id: str = ""
    reasoning_steps: int = 0
    reasoning_tokens: int = 0
    latent_steps: int = 0
    explicit_tokens: int = 0
    tool_calls: int = 0
    confidence: float = 0.0
    quality_estimate: float = 0.0
    uncertainty: float = 1.0
    marginal_quality_gain: float = 0.0
    marginal_cost: float = 0.0
    marginal_latency: float = 0.0
    marginal_compute: float = 0.0
    marginal_memory: float = 0.0
    marginal_energy: float = 0.0
    voi: float = 0.0
    expected_utility: float = 0.0
    realized_utility: float = 0.0
    budget_remaining_tokens: int = 0
    budget_remaining_cost: float = 0.0
    latency_remaining_ms: float = 0.0
    compute_remaining_flops: float = 0.0
    cost_remaining_usd: float = 0.0
    reasoning_started: bool = False
    reasoning_stopped: bool = False
    is_terminal: bool = False
    stop_reason: Optional[ReasoningStopReason] = None
    answer_history: List[str] = Field(default_factory=list)
    confidence_history: List[float] = Field(default_factory=list)
    quality_history: List[float] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    fallback_occurred: bool = False
    stagnation_count: int = 0
    execution_mode: str = "SIMULATED_LATENT_REASONING"


class ReasoningCandidate(BaseModel):
    """A candidate reasoning action with predicted economics."""
    strategy: ReasoningStrategy
    model: str
    expected_quality_gain: float = 0.0
    expected_cost_usd: float = 0.0
    expected_latency_ms: float = 0.0
    expected_compute_flops: float = 0.0
    expected_tokens: int = 0
    marginal_utility: float = 0.0
    voi_ratio: float = 0.0
    is_viable: bool = False
    reason: str = ""

    @property
    def marginal_cost_usd(self) -> float:
        return self.expected_cost_usd


class ConfidenceEstimate(BaseModel):
    """Confidence and quality tracking."""
    initial_confidence: float = 0.0
    current_confidence: float = 0.0
    confidence_gain: float = 0.0
    uncertainty: float = 1.0
    calibrated: bool = False
    quality_proxy: str = "model_self_report"
    verification_status: Optional[str] = None


class ReasoningConfig(BaseModel):
    """Phase 6 Reasoning Controller configuration."""
    reasoning_enabled: bool = True
    default_strategy: ReasoningStrategy = ReasoningStrategy.DIRECT
    max_reasoning_steps: int = 8
    max_reasoning_tokens: int = 1000
    max_latent_steps: int = 16
    confidence_target: float = 0.88
    marginal_value_threshold: float = 0.015
    reasoning_voi_threshold: float = 10.0
    reasoning_latency_budget_ms: float = 5000.0
    reasoning_cost_budget_usd: float = 0.005
    reasoning_compute_budget_flops: float = 1e13
    enable_latent: bool = True
    enable_explicit: bool = True
    enable_tool_assistance: bool = True
    enable_model_escalation: bool = True
    enable_answer_stability: bool = True
    enable_verification: bool = True
    confidence_learning_rate: float = 0.15
    stagnation_window: int = 3
    stagnation_threshold: float = 0.01
    answer_stability_window: int = 3
