"""
Memory Economics and Marginal Reuse Utility Estimator for AEON.

Models realistic cache overheads (lookup latency, bandwidth, search cost) and calculates
net utility of memory reuse versus recomputing with models.
Conforms to AEON Specification Section 11, 22, and 26.
"""

from typing import Dict, Any, Optional
from aeon.types import UtilityWeights
from aeon.memory.types import MemoryLookupResult, MemoryLevel, ReuseDecision


class MemoryEconomicsEstimator:
    """Calculates net economic utility of memory retrieval and reuse."""

    def __init__(self, weights: Optional[UtilityWeights] = None):
        self.weights = weights or UtilityWeights()

    def evaluate_memory_utility(
        self,
        lookup_res: MemoryLookupResult,
        current_difficulty: float,
    ) -> float:
        """
        Calculates Memory Net Utility:
        U_memory = ComputeSavingsValue - LookupCost - StorageCost - StalenessPenalty - QualityRisk
        """
        if not lookup_res.hit:
            # Net penalty is just the lookup overhead spent on the miss
            lookup_penalty = (
                self.weights.lambda_latency * lookup_res.lookup_latency_ms
                + self.weights.lambda_cost * lookup_res.lookup_cost_usd
            )
            return -lookup_penalty

        # Compute savings value
        savings_value = (
            self.weights.lambda_cost * lookup_res.estimated_cost_saved_usd * 10.0
            + self.weights.lambda_latency * lookup_res.estimated_latency_saved_ms * 0.5
            + self.weights.lambda_compute * lookup_res.estimated_compute_saved
        )

        # Lookup overhead costs (Realistic cache cost model - Section 26)
        lookup_penalty = (
            self.weights.lambda_latency * lookup_res.lookup_latency_ms
            + self.weights.lambda_cost * lookup_res.lookup_cost_usd
        )

        # Quality risk & verification penalty if reuse is uncertain
        quality_risk_penalty = 0.0
        if lookup_res.reuse_decision == ReuseDecision.REUSE_WITH_VERIFICATION:
            quality_risk_penalty = 0.05 * (1.0 - lookup_res.semantic_similarity)

        # Staleness penalty
        staleness_penalty = 0.0
        if lookup_res.entry and lookup_res.entry.age_seconds() > 3600.0:
            staleness_penalty = 0.02

        net_utility = savings_value - lookup_penalty - quality_risk_penalty - staleness_penalty
        return round(float(net_utility), 5)
