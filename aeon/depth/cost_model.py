"""
AEON Phase 7 — Depth & MoD Cost Model.

Computes layer-level and token-level FLOPs, latency, and monetary expenditures.
Estimates computational savings from layer skipping and early exits.
"""

from typing import Dict, Any, List, Optional
from aeon.depth.types import ModelDepthProfile, TokenComputeState
from aeon.models.model_registry import ModelRegistry, global_registry


class DepthCostModel:
    """
    Formal economic cost model for layer execution and Mixture-of-Depths compute.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or global_registry

    def calculate_layer_flops(
        self,
        model_name: str,
        total_layers: int,
        executed_layers: int,
        tokens_count: int,
    ) -> float:
        """
        Estimate FLOPs for executing N layers over T tokens.
        Formula: 2.0 * Parameters * (executed_layers / total_layers) * tokens.
        """
        if total_layers <= 0 or executed_layers <= 0 or tokens_count <= 0:
            return 0.0
        profile = self.registry.get(model_name)
        param_b = profile.param_count_b if profile else 1.5
        layer_fraction = min(1.0, max(0.0, executed_layers / float(total_layers)))
        return 2.0 * (param_b * 1e9) * layer_fraction * tokens_count

    def calculate_layer_latency_ms(
        self,
        model_name: str,
        total_layers: int,
        executed_layers: int,
        tokens_count: int,
    ) -> float:
        """
        Estimate execution latency in milliseconds for layer depth.
        """
        if total_layers <= 0 or executed_layers <= 0 or tokens_count <= 0:
            return 0.0
        profile = self.registry.get(model_name)
        base_lat = profile.base_latency_ms if profile else 25.0
        tps = max(1.0, profile.tokens_per_second if profile else 200.0)
        layer_fraction = min(1.0, max(0.0, executed_layers / float(total_layers)))
        dispatch_ms = base_lat * layer_fraction * 0.25
        gen_ms = (tokens_count * layer_fraction / tps) * 1000.0
        return round(dispatch_ms + gen_ms, 2)

    def calculate_layer_cost_usd(
        self,
        model_name: str,
        total_layers: int,
        executed_layers: int,
        tokens_count: int,
    ) -> float:
        """
        Estimate monetary cost (USD) for layer depth.
        """
        if total_layers <= 0 or executed_layers <= 0 or tokens_count <= 0:
            return 0.0
        profile = self.registry.get(model_name)
        if not profile:
            return 0.0
        blended_rate = profile.cost_per_1m_input_usd * 0.3 + profile.cost_per_1m_output_usd * 0.7
        layer_fraction = min(1.0, max(0.0, executed_layers / float(total_layers)))
        return round((tokens_count / 1_000_000.0) * blended_rate * layer_fraction, 7)

    def calculate_mod_aggregate_costs(
        self,
        model_name: str,
        total_layers: int,
        token_states: List[TokenComputeState],
    ) -> Dict[str, float]:
        """
        Aggregate compute FLOPs, latency, and monetary cost across individual token depths in MoD.
        """
        if not token_states or total_layers <= 0:
            return {"flops": 0.0, "latency_ms": 0.0, "cost_usd": 0.0, "avg_depth": 0.0}

        profile = self.registry.get(model_name)
        param_b = profile.param_count_b if profile else 1.5
        tps = max(1.0, profile.tokens_per_second if profile else 200.0)
        base_lat = profile.base_latency_ms if profile else 25.0
        blended_rate = (profile.cost_per_1m_input_usd * 0.3 + profile.cost_per_1m_output_usd * 0.7) if profile else 0.2

        total_layer_tokens = sum(t.allocated_layers for t in token_states)
        n_tokens = len(token_states)
        avg_depth = total_layer_tokens / float(n_tokens)
        overall_layer_ratio = avg_depth / float(total_layers)

        total_flops = 2.0 * (param_b * 1e9) * (total_layer_tokens / float(total_layers))
        max_token_layers = max(t.allocated_layers for t in token_states)
        pipe_latency = base_lat * (max_token_layers / float(total_layers)) * 0.25 + (total_layer_tokens / float(total_layers) / tps) * 1000.0
        total_cost = (n_tokens / 1_000_000.0) * blended_rate * overall_layer_ratio

        return {
            "flops": total_flops,
            "latency_ms": round(pipe_latency, 2),
            "cost_usd": round(total_cost, 7),
            "avg_depth": round(avg_depth, 2),
        }

    def calculate_token_flops(self, profile: ModelDepthProfile, depth: int, num_tokens: int) -> float:
        return self.calculate_layer_flops(profile.model_name, profile.total_layers, depth, num_tokens)

    def estimate_depth_latency_ms(self, profile: ModelDepthProfile, depth: int, num_tokens: int) -> float:
        return self.calculate_layer_latency_ms(profile.model_name, profile.total_layers, depth, num_tokens)

    def estimate_depth_cost_usd(self, profile: ModelDepthProfile, depth: int, num_tokens: int) -> float:
        return self.calculate_layer_cost_usd(profile.model_name, profile.total_layers, depth, num_tokens)


LayerCostModel = DepthCostModel
global_depth_cost_model = DepthCostModel()

