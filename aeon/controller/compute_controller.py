"""
AEON Compute Controller — The Central Architectural Engine.

Receives requests, analyzes complexity, allocates budgets, plans execution routes,
dispatches to inference backends, monitors confidence, and emits structured telemetry.
Conforms to AEON Specification Section 6 and Section 24.
"""

import uuid
import time
from typing import Optional, Dict, Any
from aeon.types import (
    InferenceRequest,
    InferenceResult,
    ExecutionPlan,
    Telemetry,
    TaskType,
    DifficultyLevel,
    RouteChoice,
    ReasoningMode,
)
from aeon.routing.task_classifier import TaskClassifier
from aeon.routing.model_router import ModelRouter
from aeon.routing.cascade_router import CascadeRouter
from aeon.controller.budget_allocator import BudgetAllocator
from aeon.controller.policy_engine import ComputePolicyEngine
from aeon.models.model_registry import ModelRegistry, global_registry
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.inference.simulation_backend import SimulationBackend


class ComputeController:
    """Central intelligent orchestrator for AEON adaptive inference."""

    def __init__(
        self,
        backend: Optional[BaseInferenceBackend] = None,
        registry: Optional[ModelRegistry] = None,
        policy_engine: Optional[ComputePolicyEngine] = None,
        use_cascade: bool = False,
    ):
        self.registry = registry or global_registry
        self.backend = backend or SimulationBackend(self.registry)
        self.classifier = TaskClassifier()
        self.router = ModelRouter()
        self.budget_allocator = BudgetAllocator()
        self.policy_engine = policy_engine or ComputePolicyEngine()
        self.use_cascade = use_cascade
        self.cascade_router = CascadeRouter(self.backend, self.registry)

    def plan_execution(self, request: InferenceRequest) -> ExecutionPlan:
        """Analyze request and generate structured execution plan."""
        # 1. Classify task type and estimate intrinsic difficulty
        task_type, difficulty, level, uncertainty = self.classifier.classify(request.query)

        # 2. Allocate component compute budget
        budget = self.budget_allocator.allocate(
            task_type=task_type,
            difficulty=difficulty,
            uncertainty=uncertainty,
            sla_ms=request.sla_latency_ms,
            cost_limit_usd=request.cost_budget_usd,
        )

        # 3. Determine computational route and reasoning mode
        route, reasoning_mode, needs_verification = self.router.route(
            task_type=task_type,
            difficulty=difficulty,
            level=level,
            uncertainty=uncertainty,
            cost_budget_usd=request.cost_budget_usd,
            sla_ms=request.sla_latency_ms,
        )

        selected_model = self.registry.get_model_for_route(route)

        return ExecutionPlan(
            task_type=task_type,
            difficulty=difficulty,
            difficulty_level=level,
            uncertainty=uncertainty,
            route=route,
            model=selected_model.name,
            reasoning=reasoning_mode,
            reasoning_budget=int(budget.max_tokens * budget.reasoning),
            max_compute=budget.total_compute_ratio,
            verification=needs_verification,
            cache=True,
            stop_policy="confidence" if route != RouteChoice.LARGE_MODEL else "value_of_compute",
        )

    def run(self, request: InferenceRequest) -> InferenceResult:
        """Execute end-to-end inference request with telemetry."""
        request_id = f"aeon-{uuid.uuid4().hex[:10]}"
        plan = self.plan_execution(request)

        # If cascade mode is explicitly active and not on decision model
        if self.use_cascade and plan.route != RouteChoice.DECISION_MODEL:
            res = self.cascade_router.execute_cascade(
                request=request,
                task_type=plan.task_type,
                difficulty=plan.difficulty,
                level=plan.difficulty_level,
                uncertainty=plan.uncertainty,
            )
            route_str = res["route"]
            model_name = res["final_model"]
            escalated = res.get("escalated", False)
        else:
            res = self.backend.generate(request, plan)
            route_str = plan.route.value
            model_name = plan.model
            escalated = False

        # Build telemetry conforming to Section 20
        telemetry = Telemetry(
            request_id=request_id,
            task_type=plan.task_type.value,
            route=route_str,
            model=model_name,
            reasoning_mode=plan.reasoning.value,
            input_tokens=res.get("input_tokens", 0),
            output_tokens=res.get("output_tokens", 0),
            reasoning_tokens=res.get("reasoning_tokens", 0),
            latency_ms=res.get("latency_ms", 0.0),
            compute_estimate=res.get("compute_estimate", 0.0),
            memory_mb=res.get("memory_mb", 0.0),
            estimated_energy=res.get("estimated_energy", 0.0),
            estimated_cost=res.get("estimated_cost", 0.0),
            cache_hit=False,
            speculative=res.get("speculative", False),
            verification=res.get("verification", False),
            confidence=res.get("confidence", 1.0),
            escalated=escalated,
            correct=res.get("correct"),
            latency_type="MEASURED",
            cost_type="ESTIMATED",
            compute_type="ESTIMATED",
            energy_type="ESTIMATED",
        )

        return InferenceResult(
            answer=res.get("answer", ""),
            route=route_str,
            compute=telemetry.compute_estimate,
            cost=telemetry.estimated_cost,
            latency_ms=telemetry.latency_ms,
            confidence=telemetry.confidence,
            trace=telemetry,
            plan=plan,
        )
