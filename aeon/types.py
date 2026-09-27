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


class Telemetry(BaseModel):
    """
    Structured telemetry schema conforming to AEON specification Section 20.
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
    
    # Metadata on metric types (MEASURED vs ESTIMATED vs THEORETICAL)
    latency_type: str = "MEASURED"
    cost_type: str = "ESTIMATED"
    compute_type: str = "ESTIMATED"
    energy_type: str = "ESTIMATED"


class InferenceRequest(BaseModel):
    """Request payload sent to the AEON controller."""
    query: str
    context: Optional[str] = None
    sla_latency_ms: Optional[float] = 2000.0
    cost_budget_usd: Optional[float] = 0.01
    task_hint: Optional[str] = None
    ground_truth: Optional[str] = None
    expected_answer: Optional[str] = None


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
