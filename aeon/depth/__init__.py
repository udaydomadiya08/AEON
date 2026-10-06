"""
AEON Phase 7 — Adaptive Model Depth & Mixture-of-Depths (MoD).

Public API exports for Phase 7 depth-aware computation.
"""

from aeon.depth.types import (
    DepthStrategy,
    DepthExitDecision,
    TokenImportanceSignal,
    ModelDepthProfile,
    TokenComputeState,
    DepthExecutionState,
    DepthCandidate,
    DepthConfig,
)
from aeon.depth.profiles import (
    ModelDepthRegistry,
    DEFAULT_DEPTH_PROFILES,
    global_depth_registry,
    get_depth_profile,
    get_model_depth_profile,
)
from aeon.depth.cost_model import (
    DepthCostModel,
    LayerCostModel,
    global_depth_cost_model,
)
from aeon.depth.token_importance import (
    TokenImportanceEstimator,
    global_token_importance_estimator,
)
from aeon.depth.capacity_controller import (
    CapacityController,
    global_capacity_controller,
)
from aeon.depth.mod_router import (
    MoDRouter,
    global_mod_router,
)
from aeon.depth.early_exit import (
    EarlyExitEvaluator,
    global_early_exit_evaluator,
)
from aeon.depth.economics import (
    DepthEconomics,
    global_depth_economics,
)
from aeon.depth.backend import (
    BaseDepthBackend,
    SimulationDepthBackend,
    global_simulation_depth_backend,
    global_depth_backend,
)
from aeon.depth.controller import (
    DepthController,
    global_depth_controller,
)
from aeon.depth.telemetry import build_depth_telemetry_dict

__all__ = [
    "DepthStrategy",
    "DepthExitDecision",
    "TokenImportanceSignal",
    "ModelDepthProfile",
    "TokenComputeState",
    "DepthExecutionState",
    "DepthCandidate",
    "DepthConfig",
    "ModelDepthRegistry",
    "DEFAULT_DEPTH_PROFILES",
    "global_depth_registry",
    "get_depth_profile",
    "DepthCostModel",
    "global_depth_cost_model",
    "TokenImportanceEstimator",
    "global_token_importance_estimator",
    "CapacityController",
    "global_capacity_controller",
    "MoDRouter",
    "global_mod_router",
    "EarlyExitEvaluator",
    "global_early_exit_evaluator",
    "DepthEconomics",
    "global_depth_economics",
    "BaseDepthBackend",
    "SimulationDepthBackend",
    "global_simulation_depth_backend",
    "DepthController",
    "global_depth_controller",
    "build_depth_telemetry_dict",
]
