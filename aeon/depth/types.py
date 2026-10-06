"""
AEON Phase 7 — Adaptive Model Depth & Mixture-of-Depths (MoD) Types.

Provides strongly-typed schemas for:
- Model Depth Profiles (layers, safe minimums, exit points, layer costs)
- Exit Decisions and Stopping Criteria
- Token-level Compute State and Importance
- Mixture-of-Depths (MoD) Routing and Capacity
- Depth Execution State and Structured Telemetry
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DepthExitDecision(str, Enum):
    """Early exit evaluation outcome."""
    CONTINUE = "continue"
    EXIT = "exit"
    FALLBACK_FULL_DEPTH = "fallback_full_depth"
    BUDGET_EXHAUSTED = "budget_exhausted"
    SLA_EXCEEDED = "sla_exceeded"
    QUALITY_SUFFICIENT = "quality_sufficient"
    CONFIDENCE_SUFFICIENT = "confidence_sufficient"
    VERIFICATION_FAILED = "verification_failed"
    LOW_MARGINAL_GAIN = "low_marginal_gain"
    ERROR = "error"


class DepthStrategy(str, Enum):
    """High-level model depth allocation strategy."""
    FULL_DEPTH = "full_depth"
    FIXED_EARLY_EXIT = "fixed_early_exit"
    ADAPTIVE_EARLY_EXIT = "adaptive_early_exit"
    FIXED_MOD = "fixed_mod"
    ADAPTIVE_MOD = "adaptive_mod"
    HYBRID_DEPTH = "hybrid_depth"


class TokenImportanceSignal(str, Enum):
    """Mechanism used to estimate token/position importance for MoD routing."""
    ATTENTION_PROXY = "attention_proxy"
    UNCERTAINTY = "uncertainty"
    ENTROPY_PROXY = "entropy_proxy"
    SEMANTIC_IMPORTANCE = "semantic_importance"
    UNIFORM = "uniform"


class ModelDepthProfile(BaseModel):
    """
    Depth characteristics and layer economics of a neural model tier.
    Conforms to AEON Specification for Model Depth Abstraction.
    """
    model_name: str
    total_layers: int
    min_safe_depth: int
    supported_early_exit: bool = True
    supported_mod: bool = True
    layer_cost_usd: float = 0.0
    layer_latency_ms: float = 0.0
    layer_flops: float = 0.0
    exit_points: List[int] = Field(default_factory=list)
    quality_curve: Dict[int, float] = Field(default_factory=dict)
    is_synthetic: bool = True

    @property
    def minimum_safe_depth(self) -> int:
        return self.min_safe_depth

    def get_quality_at_depth(self, depth: int) -> float:
        if depth <= 0:
            return 0.1
        if depth in self.quality_curve:
            return self.quality_curve[depth]
        if depth >= self.total_layers:
            return max(self.quality_curve.values()) if self.quality_curve else 0.95
        closest = min(self.quality_curve.keys(), key=lambda k: abs(k - depth), default=1)
        return self.quality_curve.get(closest, 0.5)



class TokenComputeState(BaseModel):
    """
    Token/position-level computational state for Mixture-of-Depths (MoD).
    Tracks individual token depth allocation and importance.
    """
    token_id: int
    token_text: str
    importance: float = 0.5  # Normalized in [0.0, 1.0]
    min_layers: int = 1
    max_layers: int = 32
    allocated_layers: int = 16
    executed_layers: int = 0
    confidence: float = 0.0
    quality_estimate: float = 0.0
    compute_flops: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    exit_status: DepthExitDecision = DepthExitDecision.CONTINUE


class DepthExecutionState(BaseModel):
    """
    Complete state tracking for a depth-optimized inference pass.
    """
    request_id: str = ""
    model_name: str = ""
    strategy: DepthStrategy = DepthStrategy.ADAPTIVE_EARLY_EXIT
    total_layers: int = 32
    requested_depth: int = 32
    actual_depth: int = 32
    min_depth: int = 8
    max_depth: int = 32
    tokens_count: int = 0
    active_tokens: int = 0
    skipped_tokens: int = 0
    deep_tokens: int = 0
    shallow_tokens: int = 0
    executed_layers: int = 0
    skipped_layers: int = 0
    avg_token_depth: float = 0.0
    layer_execution_ratio: float = 1.0
    estimated_flops: float = 0.0
    estimated_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    confidence: float = 0.0
    quality_estimate: float = 0.0
    uncertainty: float = 0.0
    exit_decision: DepthExitDecision = DepthExitDecision.CONTINUE
    stop_reason: Optional[str] = None
    early_exit_occurred: bool = False
    early_exit: bool = False
    early_exit_reason: Optional[str] = None
    mod_occurred: bool = False
    fallback_occurred: bool = False
    depth_voi: float = 0.0
    marginal_depth_gain: float = 0.0
    controller_overhead_ms: float = 0.0
    token_states: List[TokenComputeState] = Field(default_factory=list)
    action_trace: List[Dict[str, Any]] = Field(default_factory=list)
    is_simulated: bool = True
    executed_tokens: int = 0
    execution_mode: str = "SIMULATED_DEPTH_EXECUTION"


class DepthCandidate(BaseModel):
    """Candidate depth strategy evaluated by DepthEconomics."""
    strategy: DepthStrategy
    allocated_depth: int
    capacity_ratio: float = 1.0
    expected_quality: float = 0.0
    expected_quality_gain: float = 0.0
    expected_cost_usd: float = 0.0
    expected_latency_ms: float = 0.0
    expected_compute_flops: float = 0.0
    marginal_utility: float = 0.0
    voi_ratio: float = 0.0
    is_viable: bool = False
    reason: str = ""

    @property
    def estimated_cost_usd(self) -> float:
        return self.expected_cost_usd

    @property
    def estimated_latency_ms(self) -> float:
        return self.expected_latency_ms

    @property
    def estimated_compute_flops(self) -> float:
        return self.expected_compute_flops

    @property
    def estimated_quality(self) -> float:
        return self.expected_quality

    @property
    def expected_gain(self) -> float:
        return self.expected_quality_gain

    @property
    def estimated_gain(self) -> float:
        return self.expected_quality_gain


class DepthConfig(BaseModel):
    """Configuration for Phase 7 Adaptive Depth & MoD Controller."""
    depth_enabled: bool = True
    default_strategy: DepthStrategy = DepthStrategy.ADAPTIVE_EARLY_EXIT
    min_depth_ratio: float = 0.25
    max_depth_ratio: float = 1.0
    target_quality: float = 0.88
    confidence_threshold: float = 0.85
    voi_depth_threshold: float = 5.0
    mod_capacity_ratio: float = 0.65
    enable_early_exit: bool = True
    enable_mod: bool = True
    enable_token_importance: bool = True
    enable_adaptive_capacity: bool = True
    enable_verification_check: bool = True
    exit_patience: int = 1
    calibrated: bool = True

    def __init__(self, **data):
        if "strategy" in data and "default_strategy" not in data:
            data["default_strategy"] = data.pop("strategy")
        if "early_exit_enabled" in data and "enable_early_exit" not in data:
            data["enable_early_exit"] = data.pop("early_exit_enabled")
        if "mod_enabled" in data and "enable_mod" not in data:
            data["enable_mod"] = data.pop("mod_enabled")
        if "adaptive_capacity" in data and "enable_adaptive_capacity" not in data:
            data["enable_adaptive_capacity"] = data.pop("adaptive_capacity")
        if "uniform_importance" in data and "enable_token_importance" not in data:
            data["enable_token_importance"] = not data.pop("uniform_importance")
        if "voi_threshold" in data and "voi_depth_threshold" not in data:
            data["voi_depth_threshold"] = data.pop("voi_threshold")
        super().__init__(**data)
