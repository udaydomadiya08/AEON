"""
VOI-Based Budget Allocator for AEON.
Allocates inference budgets across routing, retrieval, reasoning, and generation.
Conforms to AEON Research Document Section 24.7.
"""

from pydantic import BaseModel
from typing import Optional
from aeon.types import TaskType, DifficultyLevel


class ComponentBudget(BaseModel):
    routing: float = 0.05
    retrieval: float = 0.0
    reasoning: float = 0.60
    generation: float = 0.35
    total_compute_ratio: float = 1.0
    max_tokens: int = 512


class BudgetAllocator:
    """Allocates computational budget based on task type, difficulty, and SLA constraints."""

    def allocate(
        self,
        task_type: TaskType,
        difficulty: float,
        uncertainty: float,
        sla_ms: Optional[float] = 2000.0,
        cost_limit_usd: Optional[float] = 0.01,
    ) -> ComponentBudget:
        # Base token budget
        if difficulty < 0.35:
            max_tokens = 64
            compute_ratio = 0.15
        elif difficulty < 0.70:
            max_tokens = 256
            compute_ratio = 0.50
        else:
            max_tokens = 1024
            compute_ratio = 1.0

        # Adjust for uncertainty
        if uncertainty > 0.65:
            compute_ratio = min(1.0, compute_ratio * 1.3)
            max_tokens = int(max_tokens * 1.25)
        elif uncertainty < 0.25:
            compute_ratio *= 0.8

        # Adjust for SLA latency
        if sla_ms and sla_ms < 500:
            compute_ratio *= 0.6
            max_tokens = min(max_tokens, 128)

        # Allocate across components
        if task_type == TaskType.CLASSIFICATION:
            return ComponentBudget(
                routing=0.70,
                retrieval=0.0,
                reasoning=0.0,
                generation=0.30,
                total_compute_ratio=round(compute_ratio, 3),
                max_tokens=max_tokens,
            )
        elif task_type in [TaskType.REASONING, TaskType.MATH, TaskType.CODE]:
            if difficulty > 0.70:
                return ComponentBudget(
                    routing=0.05,
                    retrieval=0.10,
                    reasoning=0.70,
                    generation=0.15,
                    total_compute_ratio=round(compute_ratio, 3),
                    max_tokens=max_tokens,
                )
            else:
                return ComponentBudget(
                    routing=0.05,
                    retrieval=0.05,
                    reasoning=0.55,
                    generation=0.35,
                    total_compute_ratio=round(compute_ratio, 3),
                    max_tokens=max_tokens,
                )
        else:  # FACTUAL, CREATIVE, EXTRACTION
            return ComponentBudget(
                routing=0.05,
                retrieval=0.20,
                reasoning=0.25,
                generation=0.50,
                total_compute_ratio=round(compute_ratio, 3),
                max_tokens=max_tokens,
            )
