"""
AEON Phase 7 — MoD Capacity Controller.

Dynamically regulates the aggregate compute capacity ratio for Mixture-of-Depths (MoD).
Ensures computation expands for complex queries and contracts under tight SLAs or small budgets.
"""

from typing import Optional, Any


class CapacityController:
    """
    Computes optimal MoD capacity ratio C in [0.20, 1.0].
    """

    def __init__(self, default_capacity: float = 0.65):
        self.default_capacity = default_capacity

    def compute_capacity(
        self,
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
        target_quality: Optional[float] = None,
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
    ) -> float:
        """
        Determine the fraction of total model depth capacity allocated across tokens.
        """
        capacity = self.default_capacity

        # Difficulty & uncertainty scaling
        if difficulty > 0.7 or uncertainty > 0.6:
            capacity += 0.20
        elif difficulty <= 0.35 or uncertainty <= 0.25:
            capacity -= 0.20

        # Quality target scaling
        if target_quality is not None:
            if target_quality >= 0.95:
                capacity += 0.15
            elif target_quality < 0.75:
                capacity -= 0.15

        # SLA / Latency constraints
        if latency_budget_ms is not None:
            if latency_budget_ms < 100.0:
                capacity = min(capacity, 0.40)
            elif latency_budget_ms < 50.0:
                capacity = min(capacity, 0.25)

        # Cost budget constraints
        if cost_budget_usd is not None:
            if cost_budget_usd < 0.0001:
                capacity = min(capacity, 0.30)

        return round(max(0.20, min(1.0, capacity)), 3)

    def determine_capacity(
        self,
        request: Any = None,
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
        target_quality: Optional[float] = None,
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
    ) -> float:
        """Alias accepting request object or individual parameters."""
        if request is not None:
            tq = target_quality or getattr(request, "target_quality", None)
            cb = cost_budget_usd or getattr(request, "cost_budget_usd", None)
            lb = latency_budget_ms or getattr(request, "sla_latency_ms", None)
        else:
            tq = target_quality
            cb = cost_budget_usd
            lb = latency_budget_ms
        return self.compute_capacity(
            difficulty=difficulty,
            uncertainty=uncertainty,
            target_quality=tq,
            cost_budget_usd=cb,
            latency_budget_ms=lb,
        )


global_capacity_controller = CapacityController()
