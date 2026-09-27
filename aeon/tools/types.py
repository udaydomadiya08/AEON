"""
AEON Phase 4 Tool Types and Contracts.

Defines the formal data models for tools, capabilities, execution results,
verification results, computational plans, and tool budgets.
Conforms to AEON Specification Section 3, 4, 5, 15, 18, 21.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from aeon.types import TaskType, ControllerAction


class ToolType(str, Enum):
    CALCULATOR = "calculator"
    PYTHON = "python"
    SYMPY = "sympy"
    RETRIEVAL = "retrieval"
    STRUCTURED = "structured"
    MEMORY = "memory"
    MODEL = "model"
    VERIFIER = "verifier"


class ToolCategory(str, Enum):
    DETERMINISTIC = "deterministic"
    PROGRAMMATIC = "programmatic"
    SYMBOLIC = "symbolic"
    RETRIEVAL = "retrieval"
    PARAMETRIC_MODEL = "parametric_model"
    MEMORY = "memory"
    VERIFICATION = "verification"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    INVALID_INPUT = "invalid_input"
    RESOURCE_LIMIT = "resource_limit"
    EXECUTION_ERROR = "execution_error"
    LOW_CONFIDENCE = "low_confidence"
    VERIFICATION_FAILED = "verification_failed"
    NETWORK_ERROR = "network_error"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_SKIPPED = "verification_skipped"


class ToolCost(BaseModel):
    """Resource and monetary cost model for tool execution."""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    compute_flops: float = 0.0
    memory_mb: float = 0.0
    energy_joules: float = 0.0


class ToolCapability(BaseModel):
    """
    Explicit capability descriptor for a computational tool.
    Informs AEON's economic candidate generator and planner.
    """
    tool_type: ToolType
    category: ToolCategory
    supported_task_types: List[TaskType]
    supported_operations: List[str] = Field(default_factory=list)
    deterministic: bool = True
    requires_network: bool = False
    requires_sandbox: bool = False
    supports_batch: bool = False
    expected_quality: float = 0.99
    expected_latency_ms: float = 0.5
    expected_cost_usd: float = 0.000001
    expected_compute_flops: float = 1e5
    expected_memory_mb: float = 1.0
    expected_energy_joules: float = 0.00001
    risk_level: float = 0.01
    confidence: float = 0.99
    execution_mode: str = "REAL_EXECUTION"  # REAL_EXECUTION, RESTRICTED_EXECUTION, REAL_SANDBOX, LOCAL_RETRIEVAL, SIMULATED


class ToolRequest(BaseModel):
    """Request payload sent to an individual tool."""
    query: str
    task_type: TaskType = TaskType.MATH
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 3.0
    max_memory_mb: float = 256.0
    require_verification: bool = False


class ToolResult(BaseModel):
    """Result emitted by a tool execution."""
    tool_type: ToolType
    status: ToolStatus
    output: Any
    error: Optional[str] = None
    actual_cost: ToolCost = Field(default_factory=ToolCost)
    confidence: float = 1.0
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Outcome of verifying a tool or model result."""
    status: VerificationStatus
    verified_output: Any
    verifier_used: str
    discrepancy_detected: bool = False
    explanation: str = ""
    verification_cost_usd: float = 0.0
    verification_latency_ms: float = 0.0


class PlanStep(BaseModel):
    """Discrete step within a multi-step computation plan."""
    step_id: int
    action: ControllerAction
    tool: str
    input_query: str
    dependencies: List[int] = Field(default_factory=list)
    expected_cost_usd: float = 0.0
    expected_latency_ms: float = 0.0
    output: Optional[Any] = None
    status: Optional[str] = None


class ComputationPlan(BaseModel):
    """Structured multi-step plan produced and optimized by the planner."""
    plan_id: str
    steps: List[PlanStep] = Field(default_factory=list)
    estimated_quality: float = 0.95
    estimated_cost_usd: float = 0.00001
    estimated_latency_ms: float = 1.0
    estimated_compute_flops: float = 1e6
    estimated_risk: float = 0.01
    requires_verification: bool = False
    verification_plan: Optional[str] = None


class ToolBudget(BaseModel):
    """Budget limits specifically for tool and programmatic executions."""
    max_tool_calls: int = 5
    max_tool_latency_ms: float = 2000.0
    max_tool_cost_usd: float = 0.005
    max_python_runtime_seconds: float = 2.0
    max_python_memory_mb: float = 128.0
    max_retrieval_calls: int = 3
    max_verification_cost_usd: float = 0.001


class ToolStats(BaseModel):
    """Cumulative operational telemetry across all tool executions."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    calls_by_tool: Dict[str, int] = Field(default_factory=dict)
    unnecessary_models_avoided: int = 0
    total_compute_saved_flops: float = 0.0
    total_cost_saved_usd: float = 0.0
    total_latency_saved_ms: float = 0.0
    verification_runs: int = 0
    verification_failures: int = 0
    fallbacks_triggered: int = 0
