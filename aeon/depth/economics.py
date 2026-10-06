"""
AEON Phase 7 — Depth Economics.

Evaluates marginal quality gains vs computational costs of layer depth.
Computes multi-objective utility, VOI ratios, and break-even thresholds.
"""

import math
from typing import List, Optional, Dict, Any
from aeon.depth.types import (
    DepthStrategy,
    DepthCandidate,
    ModelDepthProfile,
    DepthConfig,
)
from aeon.depth.cost_model import DepthCostModel, global_depth_cost_model
from aeon.types import UtilityWeights
from aeon.models.model_registry import ModelRegistry, global_registry


def _sanitize_float(
    val: Any,
    default: float = 0.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Safely clamp float against NaN, inf, or out-of-bounds values."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        if min_val is not None and f < min_val:
            return min_val
        if max_val is not None and f > max_val:
            return max_val
        return f
    except (ValueError, TypeError):
        return default


class DepthEconomics:
    """
    Economic evaluator for model depth and Mixture-of-Depths configurations.
    """

    def __init__(
        self,
        cost_model: Optional[DepthCostModel] = None,
        weights: Optional[UtilityWeights] = None,
        config: Optional[DepthConfig] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.cost_model = cost_model or global_depth_cost_model
        self.weights = weights or UtilityWeights()
        self.config = config or DepthConfig()
        self.registry = registry or global_registry

    def evaluate_candidates(
        self,
        profile: ModelDepthProfile,
        tokens_count: Optional[int] = None,
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
        cost_budget_usd: Optional[float] = None,
        latency_budget_ms: Optional[float] = None,
        request: Any = None,
        query_tokens: Optional[int] = None,
        target_quality: Optional[float] = None,
        **kwargs,
    ) -> List[DepthCandidate]:
        """
        Evaluate candidate depth strategies and rank by marginal utility.
        """
        if tokens_count is None:
            tokens_count = query_tokens if query_tokens is not None else 20
        diff = _sanitize_float(difficulty, 0.5, 0.0, 1.0)
        unc = _sanitize_float(uncertainty, 0.5, 0.0, 1.0)
        tokens = max(1, tokens_count)

        candidates: List[DepthCandidate] = []

        # 1. Full Depth Baseline
        candidates.append(
            self._evaluate_single(
                strategy=DepthStrategy.FULL_DEPTH,
                profile=profile,
                allocated_depth=profile.total_layers,
                capacity_ratio=1.0,
                tokens=tokens,
                difficulty=diff,
                uncertainty=unc,
                cost_budget=cost_budget_usd,
                latency_budget=latency_budget_ms,
            )
        )

        # 2. Fixed Early Exit (50% depth)
        if profile.supported_early_exit and self.config.enable_early_exit:
            half_depth = max(profile.min_safe_depth, profile.total_layers // 2)
            candidates.append(
                self._evaluate_single(
                    strategy=DepthStrategy.FIXED_EARLY_EXIT,
                    profile=profile,
                    allocated_depth=half_depth,
                    capacity_ratio=0.5,
                    tokens=tokens,
                    difficulty=diff,
                    uncertainty=unc,
                    cost_budget=cost_budget_usd,
                    latency_budget=latency_budget_ms,
                )
            )

        # 3. Adaptive Early Exit
        if profile.supported_early_exit and self.config.enable_early_exit:
            adaptive_depth = self._select_adaptive_depth(profile, diff, unc)
            candidates.append(
                self._evaluate_single(
                    strategy=DepthStrategy.ADAPTIVE_EARLY_EXIT,
                    profile=profile,
                    allocated_depth=adaptive_depth,
                    capacity_ratio=adaptive_depth / float(profile.total_layers),
                    tokens=tokens,
                    difficulty=diff,
                    uncertainty=unc,
                    cost_budget=cost_budget_usd,
                    latency_budget=latency_budget_ms,
                )
            )

        # 4. Fixed MoD (uniform capacity)
        if profile.supported_mod and self.config.enable_mod:
            mod_depth = int(round(profile.total_layers * self.config.mod_capacity_ratio))
            candidates.append(
                self._evaluate_single(
                    strategy=DepthStrategy.FIXED_MOD,
                    profile=profile,
                    allocated_depth=max(profile.min_safe_depth, mod_depth),
                    capacity_ratio=self.config.mod_capacity_ratio,
                    tokens=tokens,
                    difficulty=diff,
                    uncertainty=unc,
                    cost_budget=cost_budget_usd,
                    latency_budget=latency_budget_ms,
                )
            )

        # 5. Adaptive MoD
        if profile.supported_mod and self.config.enable_mod:
            adaptive_cap = min(1.0, max(0.35, 0.50 + diff * 0.35 + unc * 0.15))
            mod_depth = int(round(profile.total_layers * adaptive_cap))
            candidates.append(
                self._evaluate_single(
                    strategy=DepthStrategy.ADAPTIVE_MOD,
                    profile=profile,
                    allocated_depth=max(profile.min_safe_depth, mod_depth),
                    capacity_ratio=adaptive_cap,
                    tokens=tokens,
                    difficulty=diff,
                    uncertainty=unc,
                    cost_budget=cost_budget_usd,
                    latency_budget=latency_budget_ms,
                )
            )

        # 6. Hybrid Depth (Adaptive MoD + Early Exit)
        if profile.supported_early_exit and profile.supported_mod:
            hybrid_depth = max(profile.min_safe_depth, int(round(profile.total_layers * 0.55)))
            candidates.append(
                self._evaluate_single(
                    strategy=DepthStrategy.HYBRID_DEPTH,
                    profile=profile,
                    allocated_depth=hybrid_depth,
                    capacity_ratio=0.55,
                    tokens=tokens,
                    difficulty=diff,
                    uncertainty=unc,
                    cost_budget=cost_budget_usd,
                    latency_budget=latency_budget_ms,
                )
            )

        # Sort by marginal utility descending
        candidates.sort(key=lambda c: c.marginal_utility, reverse=True)
        return candidates

    def _select_adaptive_depth(self, profile: ModelDepthProfile, diff: float, unc: float) -> int:
        """Select appropriate layer target for adaptive early exit."""
        if diff < 0.40 and unc < 0.35:
            # Low complexity: minimum safe exit
            return profile.min_safe_depth
        elif diff < 0.65:
            # Medium complexity: early exit checkpoint near 50-60%
            checkpoints = [p for p in profile.exit_points if p >= profile.min_safe_depth]
            if checkpoints:
                mid_idx = len(checkpoints) // 2
                return checkpoints[mid_idx]
            return max(profile.min_safe_depth, int(round(profile.total_layers * 0.50)))
        else:
            # High complexity: higher depth checkpoint or full
            return max(profile.min_safe_depth, int(round(profile.total_layers * 0.85)))

    def _evaluate_single(
        self,
        strategy: DepthStrategy,
        profile: ModelDepthProfile,
        allocated_depth: int,
        capacity_ratio: float,
        tokens: int,
        difficulty: float,
        uncertainty: float,
        cost_budget: Optional[float],
        latency_budget: Optional[float],
    ) -> DepthCandidate:
        """Evaluate economic utility for a single depth strategy configuration."""
        safe_depth = max(profile.min_safe_depth, min(profile.total_layers, allocated_depth))
        curve_q = profile.quality_curve.get(safe_depth, 0.85)

        # Modulate quality by difficulty and strategy
        difficulty_penalty = difficulty * 0.08
        if strategy in [DepthStrategy.ADAPTIVE_MOD, DepthStrategy.HYBRID_DEPTH]:
            strategy_bonus = 0.02
        elif strategy == DepthStrategy.FIXED_EARLY_EXIT:
            strategy_bonus = -0.02
        else:
            strategy_bonus = 0.0

        expected_q = max(0.50, min(1.0, curve_q - difficulty_penalty + strategy_bonus))
        base_q = profile.quality_curve.get(profile.min_safe_depth, 0.65)
        expected_gain = max(0.0, expected_q - base_q)

        # Resource estimates
        flops = self.cost_model.calculate_layer_flops(profile.model_name, profile.total_layers, safe_depth, tokens)
        latency_ms = self.cost_model.calculate_layer_latency_ms(profile.model_name, profile.total_layers, safe_depth, tokens)
        cost_usd = self.cost_model.calculate_layer_cost_usd(profile.model_name, profile.total_layers, safe_depth, tokens)

        # Multi-objective utility: U = w_Q * effective_Q - resource_penalties - layer_overhead
        target_q = getattr(self.config, "target_quality", 0.85)
        if expected_q > target_q:
            effective_q = target_q + 0.10 * (expected_q - target_q)
        else:
            effective_q = expected_q

        layer_fraction = safe_depth / float(max(1, profile.total_layers))
        depth_penalty_rate = 0.25 if difficulty < 0.40 else 0.08
        layer_overhead = depth_penalty_rate * layer_fraction

        lambda_latency = max(0.0, getattr(self.weights, "lambda_latency", 0.00001))
        lambda_cost = max(0.0, getattr(self.weights, "lambda_cost", 2.0))
        lambda_compute = max(0.0, getattr(self.weights, "lambda_compute", 1e-15))
        w_q = max(0.0, getattr(self.weights, "quality", 1.0))

        resource_penalty = (
            layer_overhead
            + lambda_latency * latency_ms
            + lambda_cost * cost_usd
            + lambda_compute * flops
        )
        marginal_utility = (w_q * effective_q) - resource_penalty

        voi_ratio = (expected_gain / cost_usd) if cost_usd > 0 else 0.0

        # Check feasibility against budget limits
        is_viable = True
        reason = "economically_viable"
        if cost_budget is not None and cost_usd > cost_budget:
            is_viable = False
            reason = "cost_budget_exceeded"
        elif latency_budget is not None and latency_ms > latency_budget:
            is_viable = False
            reason = "latency_budget_exceeded"

        return DepthCandidate(
            strategy=strategy,
            allocated_depth=safe_depth,
            capacity_ratio=round(capacity_ratio, 3),
            expected_quality=round(expected_q, 4),
            expected_quality_gain=round(expected_gain, 4),
            expected_cost_usd=cost_usd,
            expected_latency_ms=latency_ms,
            expected_compute_flops=flops,
            marginal_utility=round(marginal_utility, 5),
            voi_ratio=round(voi_ratio, 2),
            is_viable=is_viable,
            reason=reason,
        )

    def compute_break_even_analysis(
        self,
        profile: ModelDepthProfile,
        tokens_count: int = 100,
    ) -> Dict[str, Any]:
        """
        Conduct break-even analysis: identify when computing full depth vs early exit
        is economically justified.
        """
        half_depth = max(profile.min_safe_depth, profile.total_layers // 2)
        full_depth = profile.total_layers

        cost_half = self.cost_model.calculate_layer_cost_usd(profile.model_name, full_depth, half_depth, tokens_count)
        cost_full = self.cost_model.calculate_layer_cost_usd(profile.model_name, full_depth, full_depth, tokens_count)
        delta_cost = cost_full - cost_half

        q_half = profile.quality_curve.get(half_depth, 0.80)
        q_full = profile.quality_curve.get(full_depth, 0.95)
        delta_q = q_full - q_half

        lambda_cost = max(0.001, getattr(self.weights, "lambda_cost", 2.0))
        w_q = max(0.001, getattr(self.weights, "quality", 1.0))

        # Break-even threshold: ΔQ_min = (λ_cost * ΔCost) / w_Q
        break_even_delta_q = (lambda_cost * delta_cost) / w_q

        return {
            "model_name": profile.model_name,
            "tokens": tokens_count,
            "half_depth": half_depth,
            "full_depth": full_depth,
            "delta_cost_usd": round(delta_cost, 7),
            "delta_quality_curve": round(delta_q, 4),
            "break_even_quality_gain": round(break_even_delta_q, 5),
            "is_full_depth_justified": delta_q >= break_even_delta_q,
            "interpretation": (
                f"Full depth is justified if expected quality gain exceeds {break_even_delta_q:.5f} "
                f"for {tokens_count} tokens."
            ),
        }

    def break_even_analysis(
        self,
        profile: ModelDepthProfile,
        tokens_count: int = 100,
        overhead_ms: float = 0.25,
        overhead_usd: float = 0.000001,
        val_per_q: float = 0.002,
    ) -> Dict[str, Any]:
        """Alias for compute_break_even_analysis."""
        return compute_break_even_analysis(profile, tokens_count, overhead_ms, overhead_usd, val_per_q)


global_depth_economics = DepthEconomics()


def compute_break_even_analysis(
    profile: ModelDepthProfile,
    tokens_count: int = 100,
    overhead_ms: float = 0.25,
    overhead_usd: float = 0.000001,
    val_per_q: float = 0.002,
) -> Dict[str, Any]:
    """Module-level economic break-even threshold calculator."""
    res = global_depth_economics.compute_break_even_analysis(profile, tokens_count)
    res["break_even_marginal_quality_early_exit"] = res.get("break_even_quality_gain", 0.001)
    res["break_even_marginal_quality_mod"] = round(res.get("break_even_quality_gain", 0.001) * 0.7, 5)
    res["overhead_ms"] = overhead_ms
    res["overhead_usd"] = overhead_usd
    return res

