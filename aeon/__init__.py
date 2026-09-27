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


from typing import Union, Optional, Dict, Any
from aeon.types import (
    InferenceRequest,
    InferenceResult,
    ExecutionPlan,
    Telemetry,
    BudgetConstraints,
    ComputeState,
    VOIResult,
    ControllerAction,
    DecisionReason,
)
from aeon.controller.compute_controller import ComputeController
from aeon.controller.dynamic_stopping import DynamicStoppingController
from aeon.controller.voi_estimator import VOIEstimator
from aeon.models.model_registry import global_registry, ModelRegistry
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.inference.simulation_backend import SimulationBackend
from aeon.inference.local_backend import LocalTransformersBackend

__version__ = "0.2.0"


class AEON:
    """
    AEON primary API facade.
    Provides simple inference interface: aeon.run(query) -> InferenceResult
    with comprehensive telemetry, routing breakdown, and compute accounting.
    Supports Phase 1 rule-based routing, cascades, and Phase 2 VOI dynamic stopping.
    """

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        policy: str = "rules",  # "rules" (Phase 1 default), "voi" (Phase 2), "cascade"
        use_cascade: bool = False,
        registry: Optional[ModelRegistry] = None,
        voi_threshold: float = 15.0,
    ):
        self.registry = registry or global_registry
        self.backend = backend or SimulationBackend(self.registry)
        self.policy = "cascade" if use_cascade else policy
        self.voi_threshold = voi_threshold

        self.voi_controller = DynamicStoppingController(
            backend=self.backend,
            registry=self.registry,
            voi_estimator=VOIEstimator(self.registry, voi_threshold=voi_threshold),
        )
        self.rules_controller = ComputeController(
            backend=self.backend,
            registry=self.registry,
            use_cascade=(self.policy == "cascade"),
        )

    def run(
        self,
        query: Union[str, InferenceRequest],
        context: Optional[str] = None,
        sla_latency_ms: Optional[float] = 2000.0,
        cost_budget_usd: Optional[float] = 0.01,
        target_quality: Optional[float] = None,
        budget: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[str] = None,
    ) -> InferenceResult:
        """Run an inference request through the AEON economically-optimized engine."""
        # Parse budget dictionary if provided
        budget_constraints = None
        if budget:
            budget_constraints = BudgetConstraints(
                max_cost_usd=budget.get("max_cost", budget.get("max_cost_usd", cost_budget_usd)),
                max_latency_ms=budget.get("max_latency", budget.get("max_latency_ms", sla_latency_ms)),
                target_quality=budget.get("target_quality", target_quality),
            )

        if isinstance(query, str):
            req = InferenceRequest(
                query=query,
                context=context,
                sla_latency_ms=sla_latency_ms,
                cost_budget_usd=cost_budget_usd,
                target_quality=target_quality,
                ground_truth=ground_truth,
                budget_constraints=budget_constraints,
            )
        else:
            req = query
            if budget_constraints and not req.budget_constraints:
                req.budget_constraints = budget_constraints

        # Dispatch based on policy
        if self.policy == "voi":
            return self.voi_controller.run(req)
        else:
            return self.rules_controller.run(req)

    def plan(self, query: str) -> ExecutionPlan:
        """Inspect the structured execution plan for a query without executing it."""
        req = InferenceRequest(query=query)
        return self.rules_controller.plan_execution(req)


__all__ = [
    "AEON",
    "InferenceRequest",
    "InferenceResult",
    "ExecutionPlan",
    "Telemetry",
    "ComputeState",
    "VOIResult",
    "ControllerAction",
    "DecisionReason",
    "BudgetConstraints",
    "ComputeController",
    "DynamicStoppingController",
    "VOIEstimator",
    "global_registry",
]
