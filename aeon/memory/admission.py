"""
Cache Admission Controller for AEON.

Evaluates economic viability before admitting an entry to memory:
Expected Net Savings = P(reuse) * ComputeAvoided - StorageCost - InvalidationRisk.
Prevents cache pollution from one-off queries.
Conforms to AEON Specification Section 10.
"""

from typing import Dict, Optional, Tuple
from aeon.types import TaskType
from aeon.memory.types import MemoryEntry, MemoryLevel


class CacheAdmissionController:
    """
    Evaluates whether an item should be admitted into cache based on expected future
    amortized compute savings versus ongoing memory footprint costs.
    """

    def __init__(
        self,
        min_net_savings_usd: float = 0.000005,
        storage_cost_per_mb_usd: float = 0.0000001,
    ):
        self.min_net_savings_usd = min_net_savings_usd
        self.storage_cost_per_mb_usd = storage_cost_per_mb_usd

    def should_admit(
        self,
        entry: MemoryEntry,
        task_type: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Calculates Net Economic Value of Caching:
        Returns: (should_admit: bool, reason: str)
        """
        # 1. Low quality or uncertain results are never admitted
        if entry.confidence < 0.65 or entry.quality_estimate < 0.60:
            return False, f"Rejected: Low confidence ({entry.confidence:.2f}) or quality ({entry.quality_estimate:.2f})"

        # 2. Estimate future reuse probability based on query patterns
        p_reuse = self._estimate_reuse_probability(entry, task_type, query_text)

        # 3. Compute cost avoided if reused
        compute_cost_avoided = entry.expected_cost_savings_usd

        # 4. Storage and bandwidth cost
        size_mb = entry.size_bytes / (1024.0 * 1024.0)
        storage_cost = size_mb * self.storage_cost_per_mb_usd

        # 5. Invalidation / staleness risk penalty
        invalidation_risk = 0.000002 if entry.freshness_policy == "STRICT" else 0.0000005

        # 6. Expected Net Savings
        expected_net_savings = (p_reuse * compute_cost_avoided) - storage_cost - invalidation_risk

        if expected_net_savings >= self.min_net_savings_usd:
            return True, f"Admitted: Expected net savings (${expected_net_savings:.7f}) >= threshold"
        else:
            return False, f"Rejected: Net savings (${expected_net_savings:.7f}) below hurdle rate (${self.min_net_savings_usd:.7f})"

    def _estimate_reuse_probability(
        self,
        entry: MemoryEntry,
        task_type: Optional[str],
        query_text: Optional[str],
    ) -> float:
        # High reuse probability: System prompts, factual lookups, deterministic math, common tools
        if entry.memory_level == MemoryLevel.L2_PREFIX:
            return 0.85
        if entry.memory_level == MemoryLevel.L3_INTERMEDIATE:
            return 0.75
        if task_type in ["classification", "factual", "math"]:
            return 0.60
        # Highly idiosyncratic queries (long, multi-step, personalized) have lower reuse probability
        if query_text and len(query_text.split()) > 40:
            return 0.15
        return 0.40
