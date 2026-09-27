"""
AEON — Adaptive Economically-Optimized Network.
Maximum-Efficiency AI Inference Architecture.
"""

from typing import Union, Optional
from aeon.types import InferenceRequest, InferenceResult, ExecutionPlan, Telemetry
from aeon.controller.compute_controller import ComputeController
from aeon.models.model_registry import global_registry, ModelRegistry
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.inference.simulation_backend import SimulationBackend
from aeon.inference.local_backend import LocalTransformersBackend

__version__ = "0.1.0"


class AEON:
    """
    AEON primary API facade.
    Provides simple inference interface: aeon.run(query) -> InferenceResult
    with comprehensive telemetry, routing breakdown, and compute accounting.
    """

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        use_cascade: bool = False,
        registry: Optional[ModelRegistry] = None,
    ):
        self.registry = registry or global_registry
        self.backend = backend or SimulationBackend(self.registry)
        self.controller = ComputeController(
            backend=self.backend,
            registry=self.registry,
            use_cascade=use_cascade,
        )

    def run(
        self,
        query: Union[str, InferenceRequest],
        context: Optional[str] = None,
        sla_latency_ms: Optional[float] = 2000.0,
        cost_budget_usd: Optional[float] = 0.01,
        ground_truth: Optional[str] = None,
    ) -> InferenceResult:
        """Run an inference request through the AEON economically-optimized engine."""
        if isinstance(query, str):
            req = InferenceRequest(
                query=query,
                context=context,
                sla_latency_ms=sla_latency_ms,
                cost_budget_usd=cost_budget_usd,
                ground_truth=ground_truth,
            )
        else:
            req = query

        return self.controller.run(req)

    def plan(self, query: str) -> ExecutionPlan:
        """Inspect the structured execution plan for a query without executing it."""
        req = InferenceRequest(query=query)
        return self.controller.plan_execution(req)


__all__ = [
    "AEON",
    "InferenceRequest",
    "InferenceResult",
    "ExecutionPlan",
    "Telemetry",
    "ComputeController",
    "global_registry",
]
