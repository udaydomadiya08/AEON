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
    ARITHMETIC = "arithmetic"
    SYMBOLIC_MATH = "symbolic_math"
    FINANCIAL = "financial"
    DATA_ANALYSIS = "data_analysis"
    RETRIEVAL = "retrieval"


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
    role: str = "TARGET"  # TARGET, DRAFT, DUAL
    architecture: str = "transformer"
    supported_devices: List[str] = Field(default_factory=lambda: ["cpu", "mps", "cuda"])
    compatible_targets: List[str] = Field(default_factory=list)


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
    # Phase 4 Tool Actions
    CALCULATOR = "CALCULATOR"
    PYTHON = "PYTHON"
    SYMPY = "SYMPY"
    RETRIEVAL = "RETRIEVAL"
    STRUCTURED_COMPUTE = "STRUCTURED_COMPUTE"
    EXECUTE_PLAN = "EXECUTE_PLAN"
    FALLBACK_REROUTE = "FALLBACK_REROUTE"
    # Phase 5 Speculation Actions
    STANDARD_DECODE = "STANDARD_DECODE"
    SPECULATE_DRAFT_MODEL = "SPECULATE_DRAFT_MODEL"
    SPECULATE_NGRAM = "SPECULATE_NGRAM"
    SPECULATE_MULTI_TOKEN = "SPECULATE_MULTI_TOKEN"
    ADAPT_DRAFT_LENGTH = "ADAPT_DRAFT_LENGTH"
    DISABLE_SPECULATION = "DISABLE_SPECULATION"
    # Phase 6 Reasoning Actions
    START_REASONING = "START_REASONING"
    CONTINUE_REASONING = "CONTINUE_REASONING"
    INCREASE_REASONING_DEPTH = "INCREASE_REASONING_DEPTH"
    REDUCE_REASONING = "REDUCE_REASONING"
    SWITCH_REASONING_MODE = "SWITCH_REASONING_MODE"
    CONVERT_TO_LATENT = "CONVERT_TO_LATENT"
    CONVERT_TO_EXPLICIT = "CONVERT_TO_EXPLICIT"
    STOP_REASONING = "STOP_REASONING"
    # Phase 7 Adaptive Depth & MoD Actions
    KEEP_DEPTH = "KEEP_DEPTH"
    INCREASE_DEPTH = "INCREASE_DEPTH"
    DECREASE_DEPTH = "DECREASE_DEPTH"
    EARLY_EXIT = "EARLY_EXIT"
    ALLOCATE_MORE_TO_TOKEN = "ALLOCATE_MORE_TO_TOKEN"
    ALLOCATE_LESS_TO_TOKEN = "ALLOCATE_LESS_TO_TOKEN"
    ENABLE_MOD = "ENABLE_MOD"
    DISABLE_MOD = "DISABLE_MOD"
    FALLBACK_FULL_DEPTH = "FALLBACK_FULL_DEPTH"


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
    # Phase 4 Tool Decision Reasons
    TOOL_DETERMINISTIC_SUFFICIENT = "TOOL_DETERMINISTIC_SUFFICIENT"
    TOOL_CHEAPEST_PATH = "TOOL_CHEAPEST_PATH"
    TOOL_VERIFIED = "TOOL_VERIFIED"
    TOOL_FAILED_REROUTE = "TOOL_FAILED_REROUTE"
    MODEL_UNNECESSARY = "MODEL_UNNECESSARY"
    MODEL_REQUIRED_FOR_REASONING = "MODEL_REQUIRED_FOR_REASONING"
    # Phase 5 Speculation Decision Reasons
    SPECULATION_VIABLE = "SPECULATION_VIABLE"
    SPECULATION_FASTER_EXPECTED = "SPECULATION_FASTER_EXPECTED"
    SPECULATION_REJECTED_LOW_ACCEPTANCE = "SPECULATION_REJECTED_LOW_ACCEPTANCE"
    SPECULATION_REJECTED_HIGH_OVERHEAD = "SPECULATION_REJECTED_HIGH_OVERHEAD"
    SPECULATION_REJECTED_NO_COMPATIBLE_DRAFT = "SPECULATION_REJECTED_NO_COMPATIBLE_DRAFT"
    SPECULATION_DETERIORATED_FALLBACK = "SPECULATION_DETERIORATED_FALLBACK"
    SPECULATION_LENGTH_REDUCED = "SPECULATION_LENGTH_REDUCED"
    SPECULATION_LENGTH_INCREASED = "SPECULATION_LENGTH_INCREASED"
    STANDARD_DECODE_PREFERRED = "STANDARD_DECODE_PREFERRED"
    # Phase 6 Reasoning Decision Reasons
    REASONING_BENEFICIAL = "REASONING_BENEFICIAL"
    REASONING_NOT_NEEDED = "REASONING_NOT_NEEDED"
    REASONING_ESCALATED = "REASONING_ESCALATED"
    REASONING_REDUCED = "REASONING_REDUCED"
    REASONING_STOPPED_CONFIDENCE = "REASONING_STOPPED_CONFIDENCE"
    REASONING_STOPPED_BUDGET = "REASONING_STOPPED_BUDGET"
    REASONING_STOPPED_STAGNATION = "REASONING_STOPPED_STAGNATION"
    REASONING_STOPPED_STABLE_ANSWER = "REASONING_STOPPED_STABLE_ANSWER"
    REASONING_STOPPED_LOW_VOI = "REASONING_STOPPED_LOW_VOI"
    # Phase 7 Adaptive Depth & MoD Decision Reasons
    DEPTH_EARLY_EXIT_VIABLE = "DEPTH_EARLY_EXIT_VIABLE"
    DEPTH_FULL_REQUIRED = "DEPTH_FULL_REQUIRED"
    DEPTH_LOW_MARGINAL_GAIN = "DEPTH_LOW_MARGINAL_GAIN"
    MOD_CAPACITY_OPTIMIZED = "MOD_CAPACITY_OPTIMIZED"
    DEPTH_BUDGET_EXHAUSTED = "DEPTH_BUDGET_EXHAUSTED"


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

    # Phase 4 Tool-Aware Computation Telemetry
    tool_used: Optional[str] = None
    tool_category: Optional[str] = None
    tool_cost_usd: float = 0.0
    tool_latency_ms: float = 0.0
    tool_compute_flops: float = 0.0
    verification_status: Optional[str] = None
    verification_cost_usd: float = 0.0
    unnecessary_model_avoided: bool = False
    fallback_occurred: bool = False
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list)

    # Phase 5 Speculative & Token-Efficient Inference Telemetry
    speculation_strategy: Optional[str] = None
    speculation_action: Optional[str] = None
    draft_model: Optional[str] = None
    draft_length: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    acceptance_rate: float = 0.0
    average_accepted_per_round: float = 0.0
    draft_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    speculation_compute_flops: float = 0.0
    expected_speedup: float = 1.0
    realized_speedup: float = 1.0
    expected_utility: float = 0.0
    realized_utility: float = 0.0
    break_even_acceptance: float = 0.0
    speculation_mode: str = "SIMULATED_SPECULATIVE_DECODING"
    speculation_rounds: int = 0

    # Phase 6 Adaptive Reasoning Telemetry
    reasoning_enabled: bool = False
    reasoning_strategy: Optional[str] = None
    reasoning_model: Optional[str] = None
    reasoning_depth_steps: int = 0
    reasoning_latent_steps: int = 0
    reasoning_explicit_tokens: int = 0
    reasoning_tool_calls: int = 0
    reasoning_quality_estimate: float = 0.0
    reasoning_uncertainty: float = 0.0
    reasoning_marginal_gain: float = 0.0
    reasoning_marginal_cost_usd: float = 0.0
    reasoning_marginal_latency_ms: float = 0.0
    reasoning_marginal_compute_flops: float = 0.0
    reasoning_voi: float = 0.0
    reasoning_expected_utility: float = 0.0
    reasoning_realized_utility: float = 0.0
    reasoning_stop_reason: Optional[str] = None
    reasoning_budget_tokens_used: int = 0
    reasoning_budget_tokens_max: int = 0
    reasoning_budget_cost_used: float = 0.0
    reasoning_budget_cost_max: float = 0.0
    reasoning_answer_stable: bool = False
    reasoning_stagnation_detected: bool = False
    reasoning_confidence_trace: List[float] = Field(default_factory=list)
    reasoning_action_trace: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_execution_mode: str = "SIMULATED_LATENT_REASONING"

    # Phase 7 Adaptive Model Depth & Mixture-of-Depths (MoD) Telemetry
    depth_enabled: bool = False
    depth_strategy: Optional[str] = None
    depth_total_layers: int = 0
    depth_requested: int = 0
    depth_executed: int = 0
    requested_depth: Optional[int] = None
    actual_depth: Optional[int] = None
    depth_layers_saved: int = 0
    depth_skipped_layers: int = 0
    depth_layer_ratio: float = 1.0
    depth_early_exit: bool = False
    early_exit: bool = False
    depth_exit_decision: Optional[str] = None
    early_exit_reason: Optional[str] = None
    depth_stop_reason: Optional[str] = None
    depth_mod_enabled: bool = False
    mod_enabled: bool = False
    depth_mod_capacity: float = 1.0
    mod_capacity: float = 1.0
    depth_active_tokens: int = 0
    mod_active_tokens: int = 0
    depth_deep_tokens: int = 0
    mod_deep_tokens: int = 0
    depth_shallow_tokens: int = 0
    mod_shallow_tokens: int = 0
    mod_skipped_tokens: int = 0
    depth_avg_token_depth: float = 0.0
    depth_marginal_gain: float = 0.0
    depth_marginal_cost_usd: float = 0.0
    depth_voi: float = 0.0
    depth_estimated_flops: float = 0.0
    depth_controller_overhead_ms: float = 0.0
    depth_execution_mode: str = "SIMULATED_DEPTH_EXECUTION"

    # Phase 8 Hardware, Device, Precision & Quantization Telemetry
    hardware_enabled: bool = False
    hardware_device: Optional[str] = None
    hardware_accelerator: Optional[str] = None
    precision_requested: Optional[str] = None
    precision_selected: Optional[str] = None
    quantization_requested: Optional[str] = None
    quantization_selected: Optional[str] = None
    quantization_status: Optional[str] = None
    memory_total_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_estimated_mb: float = 0.0
    memory_pressure_level: str = "NORMAL"
    thermal_state: str = "NOMINAL"
    energy_joules: float = 0.0
    energy_status: str = "ESTIMATED"
    hardware_utility: float = 0.0
    hardware_controller_overhead_ms: float = 0.0
    hardware_execution_mode: str = "MEASURED"
    hardware_profile_id: Optional[str] = None
    hardware_decision_reason: Optional[str] = None
    hardware_fallbacks: List[str] = Field(default_factory=list)

    @property
    def selected_precision(self) -> Optional[str]:
        return self.precision_selected


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
    tools: Optional[bool] = None
    tool_budget: Optional[Dict[str, Any]] = None
    require_verification: Optional[bool] = None
    speculation: Optional[bool] = None
    speculation_strategy: Optional[str] = None
    draft_model: Optional[str] = None
    max_draft_length: Optional[int] = None
    # Phase 6 Reasoning Request Options
    reasoning: Optional[bool] = None
    reasoning_strategy: Optional[str] = None
    reasoning_budget_tokens: Optional[int] = None
    reasoning_budget_cost: Optional[float] = None
    target_confidence: Optional[float] = None
    # Phase 7 Adaptive Depth & MoD Request Options
    adaptive_depth: Optional[bool] = None
    depth_strategy: Optional[str] = None
    max_depth: Optional[int] = None
    min_depth: Optional[int] = None
    enable_early_exit: Optional[bool] = None
    enable_mod: Optional[bool] = None
    # Phase 8 Hardware & Precision Request Options
    hardware_aware: Optional[bool] = None
    device: Optional[str] = None
    precision: Optional[str] = None
    quantization: Optional[str] = None
    memory_budget_mb: Optional[float] = None


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
    speculation_strategy: Optional[str] = None
    draft_model: Optional[str] = None
    draft_length: int = 0
    # Phase 6 Reasoning Plan Fields
    reasoning_strategy: Optional[str] = None
    reasoning_model: Optional[str] = None
    reasoning_budget_tokens: int = 0
    reasoning_depth: int = 0
    # Phase 7 Adaptive Depth Plan Fields
    depth_strategy: Optional[str] = None
    allocated_depth: int = 0
    total_model_layers: int = 0
    mod_capacity: float = 1.0
    # Phase 8 Hardware & Precision Plan Fields
    hardware_device: Optional[str] = None
    precision: Optional[str] = None
    quantization: Optional[str] = None
    hardware_execution_mode: Optional[str] = None


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

    @property
    def response(self) -> str:
        return self.answer

    @property
    def telemetry(self) -> Telemetry:
        return self.trace
