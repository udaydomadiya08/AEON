"""
AEON Phase 9 — Formal Benchmark & Ablation Subsystem.

Provides comprehensive benchmark orchestration, baselines, isolated ablations,
metric provenance, statistical analysis, and report generation.
"""

# Existing backward-compatible exports
from aeon.benchmarks.metrics import AggregateMetrics, calculate_aggregate_metrics
from aeon.benchmarks.harness import BenchmarkHarness
from aeon.benchmarks.synthetic_workload import generate_synthetic_workload

# Phase 9 core exports
from aeon.benchmarks.provenance import (
    MetricProvenance,
    MetricValue,
    ClaimClassification,
    classify_provenance_for_metric
)
from aeon.benchmarks.types import (
    ExecutionMode,
    WorkloadFamily,
    Workload,
    BenchmarkConfig,
    BenchmarkTrial,
    BenchmarkResult,
    BenchmarkRun,
    BaselineConfig,
    AblationConfig,
    ConstraintResult,
    ComparisonResult,
    BenchmarkEnvironment,
    BenchmarkManifest,
    BenchmarkReport,
    StatisticalSummary
)
from aeon.benchmarks.workload_registry import (
    WorkloadRegistry,
    global_workload_registry
)
from aeon.benchmarks.workloads import (
    create_canonical_workloads,
    evaluate_ground_truth
)
from aeon.benchmarks.baselines import get_canonical_baselines
from aeon.benchmarks.ablations import (
    get_canonical_ablations,
    get_extended_ablations,
    verify_ablation_isolation,
    AblationIsolationError
)
from aeon.benchmarks.statistics import (
    compute_statistical_summary,
    compute_paired_comparison
)
from aeon.benchmarks.reproducibility import (
    SeedManager,
    verify_runs_identical,
    compute_configuration_hash
)
from aeon.benchmarks.validation import (
    calculate_independent_utility,
    validate_controller_utility,
    evaluate_hard_constraints,
    BenchmarkInvalidError
)
from aeon.benchmarks.comparisons import (
    compute_comparison,
    compute_pareto_frontier
)
from aeon.benchmarks.hardware_matrix import HardwareMatrixRunner
from aeon.benchmarks.sensitivity import SensitivityAnalysisEngine
from aeon.benchmarks.integrity import BenchmarkIntegrityScanner
from aeon.benchmarks.executor import WorkloadExecutor
from aeon.benchmarks.repetitions import aggregate_trials_to_result
from aeon.benchmarks.runner import BenchmarkRunner
from aeon.benchmarks.export import export_benchmark_run
from aeon.benchmarks.report import generate_phase9_markdown_report

__all__ = [
    # Legacy
    "AggregateMetrics",
    "calculate_aggregate_metrics",
    "BenchmarkHarness",
    "generate_synthetic_workload",
    # Phase 9
    "MetricProvenance",
    "MetricValue",
    "ClaimClassification",
    "classify_provenance_for_metric",
    "ExecutionMode",
    "WorkloadFamily",
    "Workload",
    "BenchmarkConfig",
    "BenchmarkTrial",
    "BenchmarkResult",
    "BenchmarkRun",
    "BaselineConfig",
    "AblationConfig",
    "ConstraintResult",
    "ComparisonResult",
    "BenchmarkEnvironment",
    "BenchmarkManifest",
    "BenchmarkReport",
    "StatisticalSummary",
    "WorkloadRegistry",
    "global_workload_registry",
    "create_canonical_workloads",
    "evaluate_ground_truth",
    "get_canonical_baselines",
    "get_canonical_ablations",
    "get_extended_ablations",
    "verify_ablation_isolation",
    "AblationIsolationError",
    "compute_statistical_summary",
    "compute_paired_comparison",
    "SeedManager",
    "verify_runs_identical",
    "compute_configuration_hash",
    "calculate_independent_utility",
    "validate_controller_utility",
    "evaluate_hard_constraints",
    "BenchmarkInvalidError",
    "compute_comparison",
    "compute_pareto_frontier",
    "HardwareMatrixRunner",
    "SensitivityAnalysisEngine",
    "BenchmarkIntegrityScanner",
    "WorkloadExecutor",
    "aggregate_trials_to_result",
    "BenchmarkRunner",
    "export_benchmark_run",
    "generate_phase9_markdown_report"
]
