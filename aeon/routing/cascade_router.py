"""
Model Cascade Router with Confidence-Based Verification and Escalation.

Implements the multi-stage cascade specified in AEON Architecture Section 9:
Routes to initial tier (1B / 7B / 70B), checks confidence / verifier, and
escalates upon failure. Tracks escalation rates and operational metrics.
"""

from typing import Dict, Any, List
from aeon.types import (
    InferenceRequest,
    ExecutionPlan,
    RouteChoice,
    ReasoningMode,
    DifficultyLevel,
    TaskType,
)
from aeon.inference.backend_interface import BaseInferenceBackend
from aeon.models.model_registry import ModelRegistry, global_registry


class CascadeRouter:
    """
    Executes tiered model cascading:
    Easy -> 1B
    Medium -> 7B
    Hard -> 70B
    If confidence < threshold, escalates to higher tier.
    """

    def __init__(
        self,
        backend: BaseInferenceBackend,
        registry: ModelRegistry = global_registry,
        escalation_threshold: float = 0.75,
    ):
        self.backend = backend
        self.registry = registry
        self.escalation_threshold = escalation_threshold
        
        # Telemetry metrics
        self.total_requests = 0
        self.escalations = 0

    @property
    def escalation_rate(self) -> float:
        return (self.escalations / self.total_requests) if self.total_requests > 0 else 0.0

    def execute_cascade(
        self,
        request: InferenceRequest,
        task_type: TaskType,
        difficulty: float,
        level: DifficultyLevel,
        uncertainty: float,
    ) -> Dict[str, Any]:
        """
        Executes cascade and handles verification & escalation.
        """
        self.total_requests += 1

        # Determine initial tier
        if level == DifficultyLevel.EASY:
            initial_route = RouteChoice.SMALL_MODEL
            reasoning = ReasoningMode.DIRECT
        elif level == DifficultyLevel.MEDIUM:
            initial_route = RouteChoice.MEDIUM_MODEL
            reasoning = ReasoningMode.LATENT
        else:
            initial_route = RouteChoice.LARGE_MODEL
            reasoning = ReasoningMode.EXPLICIT

        initial_profile = self.registry.get_model_for_route(initial_route)
        plan = ExecutionPlan(
            task_type=task_type,
            difficulty=difficulty,
            difficulty_level=level,
            uncertainty=uncertainty,
            route=initial_route,
            model=initial_profile.name,
            reasoning=reasoning,
            verification=(level != DifficultyLevel.EASY),
        )

        # First stage inference
        result = self.backend.generate(request, plan)
        confidence = result.get("confidence", 1.0)

        # Check verification / confidence for escalation
        if confidence < self.escalation_threshold and initial_route != RouteChoice.LARGE_MODEL:
            # Escalate
            self.escalations += 1
            escalated_route = (
                RouteChoice.MEDIUM_MODEL
                if initial_route == RouteChoice.SMALL_MODEL and level == DifficultyLevel.EASY
                else RouteChoice.LARGE_MODEL
            )
            escalated_profile = self.registry.get_model_for_route(escalated_route)

            escalated_plan = ExecutionPlan(
                task_type=task_type,
                difficulty=difficulty,
                difficulty_level=level,
                uncertainty=uncertainty,
                route=escalated_route,
                model=escalated_profile.name,
                reasoning=ReasoningMode.EXPLICIT,
                verification=True,
            )

            escalated_result = self.backend.generate(request, escalated_plan)
            
            # Combine metrics
            combined_result = dict(escalated_result)
            combined_result["input_tokens"] += result["input_tokens"]
            combined_result["output_tokens"] += result["output_tokens"]
            combined_result["reasoning_tokens"] += result["reasoning_tokens"]
            combined_result["latency_ms"] += result["latency_ms"]
            combined_result["compute_estimate"] += result["compute_estimate"]
            combined_result["estimated_cost"] += result["estimated_cost"]
            combined_result["estimated_energy"] += result["estimated_energy"]
            combined_result["escalated"] = True
            combined_result["initial_model"] = initial_profile.name
            combined_result["final_model"] = escalated_profile.name
            combined_result["route"] = f"cascade:{initial_profile.name}->{escalated_profile.name}"
            return combined_result

        result["escalated"] = False
        result["initial_model"] = initial_profile.name
        result["final_model"] = initial_profile.name
        result["route"] = f"cascade:{initial_profile.name}"
        return result
