"""
Model Router for AEON.
Routes requests across execution tiers (Decision Model, Small, Medium, Large)
based on task characteristics and economic constraints.
"""

from typing import Tuple, Optional
from aeon.types import TaskType, DifficultyLevel, RouteChoice, ReasoningMode


class ModelRouter:
    """Dynamically routes each inference through the cheapest capable computational path."""

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold

    def route(
        self,
        task_type: TaskType,
        difficulty: float,
        level: DifficultyLevel,
        uncertainty: float,
        cost_budget_usd: Optional[float] = None,
        sla_ms: Optional[float] = None,
    ) -> Tuple[RouteChoice, ReasoningMode, bool]:
        """
        Determines the optimal computational route.

        Returns:
            (RouteChoice, ReasoningMode, needs_verification)
        """
        # 1. Non-autoregressive decision path
        if task_type == TaskType.CLASSIFICATION and level == DifficultyLevel.EASY:
            return RouteChoice.DECISION_MODEL, ReasoningMode.NONE, False

        # 2. Strict latency or cost bounds
        if cost_budget_usd is not None and cost_budget_usd < 0.0001:
            return RouteChoice.SMALL_MODEL, ReasoningMode.DIRECT, False

        # 3. Easy tasks: Small model with direct or latent reasoning
        if level == DifficultyLevel.EASY:
            if task_type in [TaskType.FACTUAL, TaskType.EXTRACTION]:
                return RouteChoice.SMALL_MODEL, ReasoningMode.DIRECT, False
            else:
                return RouteChoice.SMALL_MODEL, ReasoningMode.LATENT, False

        # 4. Medium tasks:
        # Medium reasoning/math/code tasks should use Medium model (7B)
        elif level == DifficultyLevel.MEDIUM:
            if task_type in [TaskType.REASONING, TaskType.MATH, TaskType.CODE]:
                return RouteChoice.MEDIUM_MODEL, ReasoningMode.LATENT, True
            else:
                return RouteChoice.SMALL_MODEL, ReasoningMode.LATENT, False

        # 5. Hard tasks: Large model with explicit chain of thought and verification
        else:  # HARD
            return RouteChoice.LARGE_MODEL, ReasoningMode.EXPLICIT, True
