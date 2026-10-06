"""
AEON Phase 9 — Hardware Evaluation Matrix.

Supports heterogeneous cross-hardware benchmarking across CPU, Apple Silicon MPS,
NVIDIA A100, and NVIDIA H100. Strictly distinguishes physical host measurements
from analytical reference-profile simulations.
"""

from typing import Dict, Any, List
from aeon.benchmarks.provenance import MetricProvenance
from aeon.hardware.profiles import (
    APPLE_SILICON_M_PROFILE,
    NVIDIA_A100_PROFILE,
    NVIDIA_H100_PROFILE,
    CPU_SERVER_PROFILE,
)


class HardwareMatrixRunner:
    """Evaluates benchmark configurations across heterogeneous target platforms."""

    TARGETS = ["cpu", "mps", "a100", "h100"]

    def __init__(self):
        self.profiles = {
            "cpu": CPU_SERVER_PROFILE,
            "mps": APPLE_SILICON_M_PROFILE,
            "a100": NVIDIA_A100_PROFILE,
            "h100": NVIDIA_H100_PROFILE
        }

    def get_provenance_for_target(self, target: str) -> MetricProvenance:
        """
        Determine whether the target represents a physical host measurement
        or an analytical reference-profile simulation.
        """
        target_norm = target.lower()
        import torch
        if target_norm == "cpu":
            return MetricProvenance.MEASURED
        elif target_norm == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return MetricProvenance.MEASURED
        elif target_norm in ["cuda", "a100", "h100"] and torch.cuda.is_available():
            return MetricProvenance.MEASURED
        else:
            return MetricProvenance.SIMULATED

    def get_hardware_description(self, target: str) -> str:
        """Return clear, non-deceptive description of target execution mode."""
        prov = self.get_provenance_for_target(target)
        if prov == MetricProvenance.MEASURED:
            return f"Physically measured on host ({target.upper()})"
        return f"Analytical reference-profile simulation ({target.upper()})"
