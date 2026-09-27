"""
Abstract Inference Backend Interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from aeon.types import InferenceRequest, ExecutionPlan, Telemetry


class BaseInferenceBackend(ABC):
    """Abstract base class for all AEON model inference backends."""

    @abstractmethod
    def generate(
        self,
        request: InferenceRequest,
        plan: ExecutionPlan,
    ) -> Dict[str, Any]:
        """
        Execute generation for a request according to the execution plan.

        Returns a dict containing:
        - "answer": str
        - "input_tokens": int
        - "output_tokens": int
        - "reasoning_tokens": int
        - "latency_ms": float
        - "confidence": float
        - "compute_estimate": float (FLOPs)
        - "memory_mb": float
        - "estimated_energy": float (Joules)
        - "estimated_cost": float (USD)
        - "speculative": bool
        - "verification": bool
        - "escalated": bool
        - "correct": Optional[bool]
        """
        pass
