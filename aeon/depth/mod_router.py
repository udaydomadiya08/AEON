"""
AEON Phase 7 — Mixture-of-Depths (MoD) Router.

Dynamically routes and allocates layer compute per token based on importance
and capacity constraints. Implements token-level early exits and layer skipping.
"""

from typing import List, Any, Optional
from aeon.depth.types import TokenComputeState, DepthExitDecision, ModelDepthProfile


class MoDRouter:
    """
    Allocates transformer layer depth per token according to economic importance.
    Enforces hard safety bounds: min_depth <= token_depth <= max_depth.
    """

    def route_token_depths(
        self,
        token_states: List[TokenComputeState],
        total_layers: int,
        min_depth: int,
        max_depth: int,
        capacity_ratio: float = 0.65,
    ) -> List[TokenComputeState]:
        """
        Allocate layer depth to each token under the capacity budget.
        """
        if not token_states:
            return []

        n_tokens = len(token_states)
        safe_min = max(1, min(min_depth, total_layers))
        safe_max = max(safe_min, min(max_depth, total_layers))

        # Hard invariant: aggregate layer budget
        total_layer_budget = int(round(n_tokens * safe_max * capacity_ratio))
        # Ensure budget at least covers minimum safe depth for all tokens
        min_required_budget = n_tokens * safe_min
        total_layer_budget = max(min_required_budget, total_layer_budget)

        # Check if all tokens have effectively identical importance
        importances = [t.importance for t in token_states]
        imp_range = max(importances) - min(importances)

        if imp_range < 0.05:
            # Uniform allocation
            uniform_depth = int(round(total_layer_budget / float(n_tokens)))
            uniform_depth = max(safe_min, min(safe_max, uniform_depth))
            for t in token_states:
                t.allocated_layers = uniform_depth
                t.executed_layers = uniform_depth
                t.min_layers = safe_min
                t.max_layers = safe_max
                t.exit_status = DepthExitDecision.CONTINUE if uniform_depth == safe_max else DepthExitDecision.EXIT
            return token_states

        # Non-uniform adaptive MoD routing
        # 1. Initialize all tokens to minimum safe depth
        for t in token_states:
            t.min_layers = safe_min
            t.max_layers = safe_max
            t.allocated_layers = safe_min
            t.executed_layers = safe_min

        remaining_budget = total_layer_budget - min_required_budget

        # 2. Sort tokens by importance descending to distribute surplus depth
        ranked_indices = sorted(range(n_tokens), key=lambda i: token_states[i].importance, reverse=True)

        for idx in ranked_indices:
            if remaining_budget <= 0:
                break
            tok = token_states[idx]
            # Extra depth this token can take
            headroom = safe_max - tok.allocated_layers
            if headroom <= 0:
                continue

            # Scale allocation by relative importance
            grant = min(headroom, remaining_budget, int(round(headroom * tok.importance)))
            if grant <= 0 and headroom > 0 and remaining_budget > 0:
                grant = min(headroom, remaining_budget, 1)

            tok.allocated_layers += grant
            tok.executed_layers += grant
            remaining_budget -= grant

        for t in token_states:
            # Enforce hard boundary invariants
            t.allocated_layers = max(safe_min, min(safe_max, t.allocated_layers))
            t.executed_layers = t.allocated_layers
            if t.allocated_layers < safe_max:
                t.exit_status = DepthExitDecision.EXIT
            else:
                t.exit_status = DepthExitDecision.CONTINUE

        return token_states

    def allocate_token_depths(
        self,
        tokens: List[str],
        importance_scores: List[float],
        profile: ModelDepthProfile,
        capacity_ratio: float = 0.65,
        min_depth: Optional[int] = None,
        max_depth: Optional[int] = None,
        uniform_fallback: bool = False,
    ) -> List[TokenComputeState]:
        """Construct TokenComputeState objects and allocate layer depths."""
        safe_min = min_depth if min_depth is not None else profile.min_safe_depth
        safe_max = max_depth if max_depth is not None else profile.total_layers
        states = [
            TokenComputeState(
                token_id=i,
                token_text=t,
                importance=importance_scores[i] if i < len(importance_scores) else 0.5,
                min_layers=safe_min,
                max_layers=safe_max,
                allocated_layers=safe_min,
            )
            for i, t in enumerate(tokens)
        ]
        if uniform_fallback:
            total_budget = int(round(len(tokens) * safe_max * capacity_ratio))
            per_tok = max(safe_min, min(safe_max, total_budget // max(1, len(tokens))))
            for s in states:
                s.allocated_layers = per_tok
                s.executed_layers = per_tok
            return states
        return self.route_token_depths(
            token_states=states,
            total_layers=profile.total_layers,
            min_depth=safe_min,
            max_depth=safe_max,
            capacity_ratio=capacity_ratio,
        )


global_mod_router = MoDRouter()
