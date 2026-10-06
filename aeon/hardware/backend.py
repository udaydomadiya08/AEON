"""
AEON Phase 8: Hardware Backend Abstraction & Simulation Implementation.

Defines the BaseHardwareBackend contract and the SimulationHardwareBackend,
ensuring clean separation between architecture interfaces and execution backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from aeon.hardware.types import (
    HardwareCandidate,
    ExecutionMode,
)


class BaseHardwareBackend(ABC):
    """Abstract interface defining execution on target hardware and precision."""

    @abstractmethod
    def execute(
        self,
        query: str,
        candidate: HardwareCandidate,
        tokens_count: int = 100,
    ) -> Dict[str, Any]:
        """Execute inference under candidate configuration."""
        pass


class SimulationHardwareBackend(BaseHardwareBackend):
    """
    Deterministic Level 2 simulation backend modeling hardware execution
    characteristics, latencies, FLOPs, and energy.
    """

    def execute(
        self,
        query: str,
        candidate: HardwareCandidate,
        tokens_count: int = 100,
    ) -> Dict[str, Any]:
        """Execute simulated hardware execution."""
        clean_answer = (
            f"Hardware-optimized ({candidate.device.value.upper()}/{candidate.precision.value}) "
            f"response for: {query[:60]}..."
        )

        return {
            "answer": clean_answer,
            "device": candidate.device.value,
            "precision": candidate.precision.value,
            "quantization": candidate.quantization.value,
            "latency_ms": candidate.estimated_latency_ms,
            "flops": candidate.estimated_flops,
            "cost_usd": candidate.estimated_cost_usd,
            "memory_mb": candidate.estimated_memory_mb,
            "energy_joules": candidate.estimated_energy_joules,
            "quality": candidate.estimated_quality,
            "execution_mode": ExecutionMode.SIMULATED.value,
        }
