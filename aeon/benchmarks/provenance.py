"""
AEON Phase 9 — Metric Provenance Subsystem.

Enforces strict provenance classification for every benchmark metric:
- MEASURED: Physically measured on host hardware (wall-clock time, OS stats).
- SIMULATED: Analytically computed via model execution simulator or reference profile.
- ESTIMATED: Derived from analytical heuristics (e.g. energy from power*time, memory).
- DERIVED: Formally calculated from other metrics (e.g. Quality/$, FLOPs, Speedup).
- EXTERNAL: Sourced from published benchmarks or cloud provider pricing APIs.
- UNAVAILABLE: Metric could not be computed or is not applicable.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class MetricProvenance(str, Enum):
    MEASURED = "MEASURED"
    SIMULATED = "SIMULATED"
    ESTIMATED = "ESTIMATED"
    DERIVED = "DERIVED"
    EXTERNAL = "EXTERNAL"
    UNAVAILABLE = "UNAVAILABLE"


class ClaimClassification(str, Enum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    SIMULATED = "SIMULATED"
    ESTIMATED = "ESTIMATED"
    HYPOTHESIS = "HYPOTHESIS"


class MetricValue(BaseModel):
    """Immutable, provenance-carrying metric container."""
    model_config = ConfigDict(frozen=True)

    value: float
    unit: str
    provenance: MetricProvenance
    source: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str = "VALID"
    notes: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.value} {self.unit} [{self.provenance.value}]"


def classify_provenance_for_metric(metric_name: str, execution_mode: str = "hybrid") -> MetricProvenance:
    """Determine the mandatory provenance tag for any benchmark metric."""
    name = metric_name.lower()
    
    # Strictly measured metrics on the actual host machine
    if any(k in name for k in ["controller_overhead", "host_detection", "lookup_latency", "dispatch_overhead"]):
        return MetricProvenance.MEASURED
    if "wall_clock" in name or (execution_mode == "measured" and "latency" in name):
        return MetricProvenance.MEASURED
        
    # Derived mathematical ratios
    if any(k in name for k in ["quality_per_dollar", "quality_per_second", "quality_per_tflops", "speedup", "reduction", "ratio", "break_even", "flops"]):
        return MetricProvenance.DERIVED
        
    # Estimated physical values
    if any(k in name for k in ["energy", "memory", "cost", "cloud_price"]):
        return MetricProvenance.ESTIMATED
        
    # Simulated execution metrics
    if any(k in name for k in ["quality", "accuracy", "latency", "tokens", "depth", "acceptance_rate"]):
        return MetricProvenance.SIMULATED
        
    return MetricProvenance.ESTIMATED
