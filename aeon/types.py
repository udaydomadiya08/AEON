"""
AEON Type Definitions.

Core domain data models, telemetry schemas, and hardware abstractions.
Follows the AEON research specification for maximum-efficiency AI inference.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    FACTUAL = "factual"
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    CREATIVE = "creative"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReasoningMode(str, Enum):
    NONE = "none"
    DIRECT = "direct"
    LATENT = "latent"
    EXPLICIT = "explicit"
    ADAPTIVE = "adaptive"


class RouteChoice(str, Enum):
    DECISION_MODEL = "decision_model"
    SMALL_MODEL = "small_model"
    MEDIUM_MODEL = "medium_model"
    LARGE_MODEL = "large_model"
    TOOL = "tool"


class HardwareProfile(BaseModel):
    """Hardware capabilities abstraction."""
    device: str = "cpu"  # cpu, mps, cuda
    vram_gb: float = 8.0
    memory_bandwidth_gbps: float = 100.0
    compute_capability: str = "arm64"
    supported_precisions: List[str] = Field(default_factory=lambda: ["FP32", "FP16", "INT8"])
    backend: str = "local"


class ModelProfile(BaseModel):
    """Model operational and economic characteristics."""
    name: str
    param_count_b: float
    precision: str = "FP16"
    cost_per_1m_input_usd: float
    cost_per_1m_output_usd: float
    base_latency_ms: float
    tokens_per_second: float
    quality_score: float  # Expected relative accuracy/capability (0.0 to 1.0)
    min_memory_mb: float
    flops_per_token: float
    is_decision_model: bool = False


class ControllerAction(str, Enum):
    STOP = "STOP"
    CONTINUE = "CONTINUE"
    ESCALATE_MODEL = "ESCALATE_MODEL"
    INCREASE_REASONING = "INCREASE_REASONING"
    VERIFY = "VERIFY"
    SWITCH_MODEL = "SWITCH_MODEL"
    RETRIEVE = "RETRIEVE"
    USE_TOOL = "USE_TOOL"
    CACHE_HIT = "CACHE_HIT"
    PARTIAL_CACHE_REUSE = "PARTIAL_CACHE_REUSE"
    RECOMPUTE = "RECOMPUTE"
    VERIFY_CACHE = "VERIFY_CACHE"


class DecisionReason(str, Enum):
    STOP_HIGH_CONFIDENCE = "STOP_HIGH_CONFIDENCE"
    STOP_LOW_VOI = "STOP_LOW_VOI"
    STOP_BUDGET_EXHAUSTED = "STOP_BUDGET_EXHAUSTED"
    STOP_MAX_ITERATIONS = "STOP_MAX_ITERATIONS"
    STOP_TARGET_QUALITY_REACHED = "STOP_TARGET_QUALITY_REACHED"
    CONTINUE_HIGH_VOI = "CONTINUE_HIGH_VOI"
    ESCALATE_HIGH_EXPECTED_GAIN = "ESCALATE_HIGH_EXPECTED_GAIN"
    VERIFY_HIGH_UNCERTAINTY = "VERIFY_HIGH_UNCERTAINTY"
    TOOL_MORE_EFFICIENT = "TOOL_MORE_EFFICIENT"
    CACHE_REUSE_MORE_EFFICIENT = "CACHE_REUSE_MORE_EFFICIENT"
    CACHE_HIT_SAFE = "CACHE_HIT_SAFE"
    PARTIAL_PREFIX_REUSE = "PARTIAL_PREFIX_REUSE"
    CACHE_REJECTED_STALE = "CACHE_REJECTED_STALE"
    CACHE_REJECTED_RISK = "CACHE_REJECTED_RISK"


class UtilityWeights(BaseModel):
    """
    Formal AEON Utility Objective Weights:
    Utility = quality * Quality - λ_latency * Latency - λ_cost * Cost - λ_compute * FLOPs - λ_memory * Memory - λ_energy * Energy
    Conforms to AEON Specification Section 4.
    """
    quality: float = Field(default=1.0, description="Multiplier for solution quality")
    lambda_latency: float = Field(default=0.00001, description="Penalty per millisecond of latency")
    lambda_cost: float = Field(default=2.0, description="Penalty per USD of monetary cost")
    lambda_compute: float = Field(default=1e-15, description="Penalty per FLOP of compute")
    lambda_memory: float = Field(default=0.000001, description="Penalty per MB of memory footprint")
    lambda_energy: float = Field(default=0.00002, description="Penalty per Joule of energy")


class BudgetConstraints(BaseModel):
    """Execution budget and constraint configuration for AEON inference."""
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    max_compute_flops: Optional[float] = None
    max_tokens: Optional[int] = None
    target_quality: Optional[float] = None
    mode: str = "balanced"  # fixed, latency-constrained, cost-constrained, quality-constrained, balanced


class Telemetry(BaseModel):
    """
    Structured telemetry schema conforming to AEON specification Section 20 and Section 16.
    Every inference MUST produce this telemetry.
    """
    request_id: str
    task_type: str
    route: str
    model: str
    reasoning_mode: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    compute_estimate: float = 0.0  # FLOPs or normalized compute units
    memory_mb: float = 0.0
    estimated_energy: float = 0.0  # Joules
    estimated_cost: float = 0.0  # USD
    cache_hit: bool = False
    speculative: bool = False
    verification: bool = False
    confidence: float = 1.0
    escalated: bool = False
    correct: Optional[bool] = None
    
    # Phase 2 VOI & Dynamic Stopping Telemetry
    initial_quality_estimate: Optional[float] = None
    final_quality_estimate: Optional[float] = None
    marginal_quality_gain: Optional[float] = None
    voi: Optional[float] = None
    marginal_cost: Optional[float] = None
    marginal_latency: Optional[float] = None
    marginal_compute: Optional[float] = None
    decision_reason: Optional[str] = None
    stopping_reason: Optional[str] = None
    budget_before: Optional[float] = None
    budget_after: Optional[float] = None
    number_of_controller_iterations: int = 1
    number_of_escalations: int = 0
    trace_steps: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata on metric types (MEASURED vs ESTIMATED vs THEORETICAL)
    latency_type: str = "MEASURED"
    cost_type: str = "ESTIMATED"
    compute_type: str = "ESTIMATED"
    energy_type: str = "ESTIMATED"

    # Phase 3 Memory Hierarchy Telemetry
    memory_hit: bool = False
    memory_level: Optional[str] = None
    memory_action: Optional[str] = None
    reused_tokens: int = 0
    compute_saved_flops: float = 0.0
    latency_saved_ms: float = 0.0
    cost_saved_usd: float = 0.0
    energy_saved_joules: float = 0.0
    lookup_latency_ms: float = 0.0
    memory_similarity: Optional[float] = None
    freshness_score: Optional[float] = None
    reuse_confidence: Optional[float] = None
    reuse_decision: Optional[str] = None
    memory_bytes_reused: int = 0


class ComputeState(BaseModel):
    """
    Explicit controller state representing execution progress for a request.
    Conforms to AEON Specification Section 6.
    """
    request_id: str
    task_type: TaskType
    difficulty: float
    difficulty_level: DifficultyLevel
    uncertainty: float
    current_model: str
    current_route: RouteChoice
    current_reasoning_mode: ReasoningMode
    current_quality_estimate: float = 0.0
    confidence: float = 0.0
    compute_spent: float = 0.0
    latency_spent_ms: float = 0.0
    monetary_cost_usd: float = 0.0
    energy_spent_joules: float = 0.0
    tokens_generated: int = 0
    reasoning_steps: int = 0
    remaining_cost_budget: Optional[float] = None
    remaining_latency_ms: Optional[float] = None
    iteration: int = 0
    is_terminal: bool = False
    decision_reason: Optional[DecisionReason] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)


class VOIResult(BaseModel):
    """Value-of-Information evaluation outcome for a candidate computational action."""
    action: ControllerAction
    expected_current_quality: float
    expected_future_quality: float
    marginal_quality_gain: float
    marginal_cost_usd: float
    marginal_latency_ms: float
    marginal_compute_flops: float
    marginal_utility: float
    voi_ratio: float
    voi_threshold: float
    is_economically_viable: bool
    reason: DecisionReason
    next_model: Optional[str] = None
    next_reasoning_mode: Optional[ReasoningMode] = None


class InferenceRequest(BaseModel):
    """Request payload sent to the AEON controller."""
    query: str
    context: Optional[str] = None
    sla_latency_ms: Optional[float] = 2000.0
    cost_budget_usd: Optional[float] = 0.01
    target_quality: Optional[float] = None
    task_hint: Optional[str] = None
    ground_truth: Optional[str] = None
    expected_answer: Optional[str] = None
    budget_constraints: Optional[BudgetConstraints] = None
    memory: Optional[bool] = None
    memory_levels: Optional[List[str]] = None
    session_id: Optional[str] = None
    model_version: Optional[str] = None
    consistency: Optional[str] = None


class ExecutionPlan(BaseModel):
    """Structured execution plan produced by the AEON compute controller."""
    task_type: TaskType
    difficulty: float = Field(ge=0.0, le=1.0)
    difficulty_level: DifficultyLevel
    uncertainty: float = Field(ge=0.0, le=1.0)
    route: RouteChoice
    model: str
    reasoning: ReasoningMode
    reasoning_budget: int = 0
    tool: Optional[str] = None
    max_compute: float = 1.0
    verification: bool = False
    cache: bool = True
    stop_policy: str = "confidence"


class InferenceResult(BaseModel):
    """Result returned by AEON."""
    answer: str
    route: str
    compute: float
    cost: float
    latency_ms: float
    confidence: float
    trace: Telemetry
    plan: Optional[ExecutionPlan] = None
