"""
AEON Phase 7 — Depth Execution Backends.

Provides:
1. BaseDepthBackend: Abstract interface for real PyTorch/llama.cpp layer-skipping hooks.
2. SimulationDepthBackend: Deterministic, reproducible simulation backend.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from aeon.depth.types import (
    TokenComputeState,
    ModelDepthProfile,
    DepthExecutionState,
    DepthStrategy,
)
from aeon.depth.cost_model import DepthCostModel, global_depth_cost_model


class BaseDepthBackend:
    """
    Base interface for model execution with early exit and Mixture-of-Depths.
    Designed for future physical transformer backends (vLLM, llama.cpp, custom CUDA kernels).
    """

    def execute_layer(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a single transformer layer forward pass."""
        raise NotImplementedError("UNSUPPORTED_REAL_BACKEND: Real GPU layer execution not attached")

    def early_exit(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Project intermediate hidden states to vocabulary for early exit."""
        raise NotImplementedError("UNSUPPORTED_REAL_BACKEND: Early exit projection head not attached")

    def execute_mod(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute mixture-of-depths token-wise compute allocation."""
        raise NotImplementedError("UNSUPPORTED_REAL_BACKEND: Token-level layer gating not attached")


class SimulationDepthBackend(BaseDepthBackend):
    """
    Deterministic simulation backend for Phase 7 Adaptive Depth & MoD.
    Accurately computes FLOPs, latency, and costs based on layer profiles.
    Conforms to Level 2 (Simulation) disclosure requirements.
    """

    def __init__(self, cost_model: Optional[DepthCostModel] = None):
        self.cost_model = cost_model or global_depth_cost_model
        self.execution_mode = "SIMULATED_DEPTH_EXECUTION"

    def execute_layer(self, layer_idx: int, hidden_states: Any, attention_mask: Optional[Any] = None) -> Any:
        raise NotImplementedError("UNSUPPORTED_REAL_BACKEND: Simulation backend does not hold physical tensor weights")

    def early_exit(self, current_layer: int, hidden_states: Any) -> Dict[str, Any]:
        raise NotImplementedError("UNSUPPORTED_REAL_BACKEND: Simulation backend does not hold physical tensor weights")

    def execute_mod(
        self,
        query: str,
        profile: ModelDepthProfile,
        token_states: List[TokenComputeState],
    ) -> Dict[str, Any]:
        """
        Simulate token-wise mixture-of-depths execution.
        """
        costs = self.cost_model.calculate_mod_aggregate_costs(
            model_name=profile.model_name,
            total_layers=profile.total_layers,
            token_states=token_states,
        )

        n_tokens = len(token_states)
        deep_tokens = sum(1 for t in token_states if t.allocated_layers >= profile.total_layers * 0.8)
        shallow_tokens = sum(1 for t in token_states if t.allocated_layers <= profile.min_safe_depth)

        # Generate a simulated clean answer based on query
        answer = self._generate_simulated_answer(query, profile.model_name)

        return {
            "answer": answer,
            "flops": costs["flops"],
            "latency_ms": costs["latency_ms"],
            "cost_usd": costs["cost_usd"],
            "avg_depth": costs["avg_depth"],
            "deep_tokens": deep_tokens,
            "shallow_tokens": shallow_tokens,
            "tokens_count": n_tokens,
            "execution_mode": self.execution_mode,
        }

    def execute_uniform_depth(
        self,
        query: str,
        profile: ModelDepthProfile,
        executed_layers: int,
        tokens_count: int,
    ) -> Dict[str, Any]:
        """
        Simulate uniform layer execution (standard or early exit).
        """
        safe_layers = max(profile.min_safe_depth, min(profile.total_layers, executed_layers))
        flops = self.cost_model.calculate_layer_flops(
            profile.model_name, profile.total_layers, safe_layers, tokens_count
        )
        latency_ms = self.cost_model.calculate_layer_latency_ms(
            profile.model_name, profile.total_layers, safe_layers, tokens_count
        )
        cost_usd = self.cost_model.calculate_layer_cost_usd(
            profile.model_name, profile.total_layers, safe_layers, tokens_count
        )

        answer = self._generate_simulated_answer(query, profile.model_name)

        return {
            "answer": answer,
            "flops": flops,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "executed_layers": safe_layers,
            "tokens_count": tokens_count,
            "execution_mode": self.execution_mode,
        }

    def _generate_simulated_answer(self, query: str, model_name: str) -> str:
        """Produce clean direct answer without internal annotations."""
        q_lower = query.lower()
        if "capital" in q_lower and "france" in q_lower:
            return "Paris is the capital of France."
        if "capital" in q_lower and "italy" in q_lower:
            return "Rome is the capital of Italy."
        if "17.3%" in q_lower and "847" in q_lower:
            return "146.531 (17.3% of 847 = 0.173 * 847 = 146.531)"
        if "sentiment" in q_lower:
            return "positive" if any(w in q_lower for w in ["great", "good", "exceeded", "love"]) else "negative"
        return f"Depth-optimized answer for: {query.strip()[:60]}..."

    def simulate_depth_run(
        self,
        profile: ModelDepthProfile,
        tokens: List[str],
        strategy: DepthStrategy = DepthStrategy.ADAPTIVE_EARLY_EXIT,
        initial_depth: Optional[int] = None,
        target_quality: float = 0.85,
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
    ) -> DepthExecutionState:
        """Simulate an end-to-end depth execution pass and return DepthExecutionState."""
        tokens_count = max(1, len(tokens))
        req_depth = initial_depth if (initial_depth is not None and initial_depth > 0) else profile.total_layers

        if strategy == DepthStrategy.FULL_DEPTH:
            actual_depth = profile.total_layers
            early_exit = False
            exit_reason = "full_depth_strategy"
        elif strategy == DepthStrategy.FIXED_EARLY_EXIT:
            actual_depth = max(profile.min_safe_depth, profile.total_layers // 2)
            early_exit = True
            exit_reason = "fixed_early_exit_50pct"
        elif strategy in (DepthStrategy.ADAPTIVE_EARLY_EXIT, DepthStrategy.HYBRID_DEPTH):
            diff_ratio = (difficulty * 0.6 + uncertainty * 0.4)
            scaled = int(round(profile.min_safe_depth + (profile.total_layers - profile.min_safe_depth) * diff_ratio))
            actual_depth = max(profile.min_safe_depth, min(profile.total_layers, scaled))
            early_exit = actual_depth < profile.total_layers
            exit_reason = "adaptive_quality_sufficient" if early_exit else "reached_full_depth"
        elif strategy in (DepthStrategy.FIXED_MOD, DepthStrategy.ADAPTIVE_MOD):
            actual_depth = max(profile.min_safe_depth, int(profile.total_layers * 0.65))
            early_exit = False
            exit_reason = "mod_capacity_allocated"
        else:
            actual_depth = profile.total_layers
            early_exit = False
            exit_reason = "default_full_depth"

        flops = self.cost_model.calculate_layer_flops(profile.model_name, profile.total_layers, actual_depth, tokens_count)
        lat = self.cost_model.calculate_layer_latency_ms(profile.model_name, profile.total_layers, actual_depth, tokens_count)
        cost = self.cost_model.calculate_layer_cost_usd(profile.model_name, profile.total_layers, actual_depth, tokens_count)
        q_est = profile.get_quality_at_depth(actual_depth)

        return DepthExecutionState(
            model_name=profile.model_name,
            strategy=strategy,
            total_layers=profile.total_layers,
            requested_depth=req_depth,
            actual_depth=actual_depth,
            min_depth=profile.min_safe_depth,
            early_exit=early_exit,
            early_exit_reason=exit_reason,
            executed_tokens=tokens_count,
            estimated_flops=flops,
            estimated_latency_ms=lat,
            estimated_cost_usd=cost,
            quality_estimate=q_est,
            confidence=0.85,
            is_simulated=True,
            execution_mode=self.execution_mode,
        )



global_simulation_depth_backend = SimulationDepthBackend()
global_depth_backend = global_simulation_depth_backend

