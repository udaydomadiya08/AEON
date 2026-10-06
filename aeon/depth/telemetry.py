"""
AEON Phase 7 — Depth Telemetry Builder.

Constructs structured telemetry records for adaptive model depth,
early exit decisions, and Mixture-of-Depths (MoD) allocations.
Preserves privacy by omitting private internal token texts.
"""

from typing import Dict, Any
from aeon.depth.types import DepthExecutionState, ModelDepthProfile


def build_depth_telemetry_dict(
    state: DepthExecutionState,
    profile: ModelDepthProfile,
    overhead_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Build structured dictionary representation of Phase 7 execution metrics.
    """
    return {
        "depth_enabled": True,
        "depth_strategy": state.strategy.value,
        "depth_total_layers": profile.total_layers,
        "depth_requested": state.requested_depth,
        "depth_executed": state.actual_depth,
        "depth_skipped_layers": state.skipped_layers,
        "depth_layer_ratio": round(state.layer_execution_ratio, 4),
        "depth_early_exit": state.early_exit_occurred,
        "depth_exit_decision": state.exit_decision.value,
        "depth_stop_reason": state.stop_reason,
        "depth_mod_enabled": state.mod_occurred,
        "depth_mod_capacity": round(state.avg_token_depth / float(profile.total_layers), 3) if profile.total_layers > 0 else 1.0,
        "depth_active_tokens": state.active_tokens,
        "depth_deep_tokens": state.deep_tokens,
        "depth_shallow_tokens": state.shallow_tokens,
        "depth_avg_token_depth": round(state.avg_token_depth, 2),
        "depth_marginal_gain": round(state.marginal_depth_gain, 4),
        "depth_voi": round(state.depth_voi, 2),
        "depth_controller_overhead_ms": round(overhead_ms, 3),
        "depth_execution_mode": state.execution_mode,
        "action_trace": state.action_trace,
    }
