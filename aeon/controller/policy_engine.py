"""
Configurable Compute Decision Policy Engine.

Evaluates multi-objective utility functions:
Utility = Capability - λ_lat * Latency - λ_cost * Cost - λ_energy * Energy - λ_mem * Memory - λ_compute * FLOPs
Conforms to AEON Specification Section 7.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class PolicyWeights(BaseModel):
    """Configurable penalty multipliers for multi-objective optimization."""
    lambda_latency: float = Field(default=0.001, description="Weight for latency penalty (per ms)")
    lambda_cost: float = Field(default=100.0, description="Weight for monetary cost penalty (per USD)")
    lambda_energy: float = Field(default=0.005, description="Weight for energy penalty (per Joule)")
    lambda_memory: float = Field(default=0.00001, description="Weight for memory footprint penalty (per MB)")
    lambda_compute: float = Field(default=1e-13, description="Weight for FLOPs penalty")


class ComputePolicyEngine:
    """Calculates utility trade-offs across candidate execution plans."""

    def __init__(self, weights: PolicyWeights = None):
        self.weights = weights or PolicyWeights()

    def evaluate_utility(
        self,
        capability: float,
        latency_ms: float,
        cost_usd: float,
        energy_joules: float,
        memory_mb: float,
        flops: float,
    ) -> float:
        """
        Computes formal utility:
        U = Capability - λ_lat*Lat - λ_cost*Cost - λ_energy*Energy - λ_mem*Memory - λ_compute*FLOPs
        """
        penalty = (
            self.weights.lambda_latency * latency_ms
            + self.weights.lambda_cost * cost_usd
            + self.weights.lambda_energy * energy_joules
            + self.weights.lambda_memory * memory_mb
            + self.weights.lambda_compute * flops
        )
        return capability - penalty
