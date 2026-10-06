"""
AEON Phase 9 — Core Benchmark Data Models.

Defines immutable, validated data models for benchmarks, workloads,
baselines, ablations, metrics, statistics, and manifests.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

from aeon.benchmarks.provenance import MetricProvenance, MetricValue, ClaimClassification


class ExecutionMode(str, Enum):
    SIMULATED = "simulated"
    MEASURED = "measured"
    HYBRID = "hybrid"


class WorkloadFamily(str, Enum):
    DETERMINISTIC_ARITHMETIC = "deterministic_arithmetic"
    SIMPLE_FACTUAL = "simple_factual"
    CLASSIFICATION = "classification"
    SIMPLE_REASONING = "simple_reasoning"
    MEDIUM_REASONING = "medium_reasoning"
    HARD_REASONING = "hard_reasoning"
    VERIFICATION_SENSITIVE = "verification_sensitive"
    LONG_CONTEXT = "long_context"
    REPEATED_QUERY = "repeated_query"
    TOOL_SUFFICIENT = "tool_sufficient"
    MODEL_REQUIRED = "model_required"
    MIXED = "mixed"


class Workload(BaseModel):
    """Immutable workload definition with strict ground truth specifications."""
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    family: WorkloadFamily
    query: str
    context: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_numeric_value: Optional[float] = None
    accepted_answer_set: Optional[List[str]] = None
    verification_rule: Optional[str] = None
    require_verification: bool = False
    
    # Non-circular difficulty taxonomy
    declared_difficulty: float = Field(ge=0.0, le=1.0)
    estimated_difficulty: float = Field(ge=0.0, le=1.0)
    observed_difficulty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def difficulty(self) -> float:
        """Declared difficulty property for backwards compatibility."""
        return self.declared_difficulty


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark execution."""
    model_config = ConfigDict(frozen=True)

    mode: ExecutionMode = ExecutionMode.HYBRID
    seed: int = 42
    repetitions: int = Field(default=1, ge=1)
    sla_ms: float = 8000.0
    cost_budget_usd: float = 0.05
    memory_budget_mb: Optional[float] = None
    target_quality: Optional[float] = None
    hardware_target: Optional[str] = None
    device_override: Optional[str] = None
    precision_override: Optional[str] = None
    quantization_override: Optional[str] = None
    strict_provenance: bool = True


class StatisticalSummary(BaseModel):
    """Statistical aggregations over repeated measurements."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    ci_95_lower: Optional[float] = None
    ci_95_upper: Optional[float] = None
    sample_size: int = 1


class BenchmarkTrial(BaseModel):
    """Result of an individual evaluation trial."""
    trial_id: int
    workload_id: str
    config_name: str
    correct: bool
    latency_ms: float
    cost_usd: float
    tokens: int
    flops: float
    memory_mb: float
    energy_joules: float
    utility: float
    depth_layers: int = 32
    device: str = "cpu"
    precision: str = "fp32"
    quantization: str = "none"
    controller_overhead_ms: float = 0.0
    metrics: Dict[str, MetricValue] = Field(default_factory=dict)
    telemetry: Optional[Dict[str, Any]] = None


class BaselineConfig(BaseModel):
    """Standardized baseline configuration."""
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    settings: Dict[str, Any]


class AblationConfig(BaseModel):
    """Isolated ablation configuration with explicit isolation diff."""
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    target_component: str
    changed_parameters: Dict[str, Any]
    unchanged_parameters: Dict[str, Any]


class ConstraintResult(BaseModel):
    """Hard-constraint validation result."""
    constraint_name: str
    threshold: float
    actual: float
    passed: bool
    is_hard: bool = True
    infeasible: bool = False


class ComparisonResult(BaseModel):
    """Direct comparative results between candidate and baseline."""
    baseline_name: str
    candidate_name: str
    delta_quality: float
    delta_latency: float
    delta_cost: float
    delta_tokens: int
    delta_flops: float
    delta_energy: float
    delta_quality_per_dollar: float
    latency_reduction_pct: float
    cost_reduction_pct: float
    token_reduction_pct: float
    is_pareto_superior: bool = False


class BenchmarkEnvironment(BaseModel):
    """Complete environment metadata for audit and reproducibility."""
    os: str
    python_version: str
    cpu: str
    ram_gb: float
    gpu: str
    git_commit: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    aeon_version: str = "0.9.0"


class BenchmarkManifest(BaseModel):
    """Cryptographically auditable manifest."""
    manifest_id: str
    environment: BenchmarkEnvironment
    config_hash: str
    workload_hash: str
    seeds: Dict[str, int]
    execution_mode: ExecutionMode
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class BenchmarkResult(BaseModel):
    """Aggregated benchmark outcome for a specific condition/baseline."""
    condition_name: str
    num_requests: int
    quality: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    mean_cost_usd: float
    total_tokens: int
    mean_flops: float
    mean_memory_mb: float
    mean_energy_joules: float
    mean_utility: float
    quality_per_dollar: float
    quality_per_second: float
    quality_per_tflops: float
    mean_depth: float
    selected_device: str
    selected_precision: str
    quantization: str
    controller_overhead_ms: float
    overhead_pct: float
    break_even_requests: float = 1.0
    provenance_summary: Dict[str, str] = Field(default_factory=dict)
    statistical_metrics: Dict[str, StatisticalSummary] = Field(default_factory=dict)
    saturation_warnings: List[str] = Field(default_factory=list)


class BenchmarkRun(BaseModel):
    """A full benchmark execution containing manifest and all results."""
    run_id: str
    manifest: BenchmarkManifest
    trials: List[BenchmarkTrial]
    results: Dict[str, BenchmarkResult]
    comparisons: Dict[str, ComparisonResult] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """Formal rendered benchmark report artifact."""
    title: str
    executive_summary: str
    markdown_content: str
    manifest: BenchmarkManifest
    results: List[BenchmarkResult]
    claims: List[Dict[str, Any]] = Field(default_factory=list)
