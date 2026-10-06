"""
AEON Phase 7 — Model Depth Profiles.

Defines layer-level operational profiles, exit points, and depth-quality
curves for each model tier in the AEON hierarchy.
"""

from typing import Dict, Optional, List
from aeon.depth.types import ModelDepthProfile
from aeon.models.model_profiles import DEFAULT_PROFILES


def _generate_depth_quality_curve(total_layers: int, max_quality: float, min_quality: float) -> Dict[int, float]:
    """
    Generate a realistic diminishing-returns depth-quality curve.
    Early layers produce rapid representation growth; later layers refine edge cases.
    """
    curve: Dict[int, float] = {}
    for layer in range(1, total_layers + 1):
        ratio = layer / float(total_layers)
        # Power-law saturation curve: Q(l) = Q_min + (Q_max - Q_min) * (1 - (1 - ratio)^1.8)
        q = min_quality + (max_quality - min_quality) * (1.0 - (1.0 - ratio) ** 1.8)
        curve[layer] = round(q, 4)
    return curve


DECISION_MODEL_DEPTH_PROFILE = ModelDepthProfile(
    model_name="aeon-decision-jev",
    total_layers=4,
    min_safe_depth=4,
    supported_early_exit=False,
    supported_mod=False,
    layer_cost_usd=0.02 / 4.0,
    layer_latency_ms=5.0 / 4.0,
    layer_flops=1.6e8 / 4.0,
    exit_points=[4],
    quality_curve=_generate_depth_quality_curve(4, 0.82, 0.65),
    is_synthetic=True,
)

DRAFT_0_5B_DEPTH_PROFILE = ModelDepthProfile(
    model_name="aeon-draft-0.5b",
    total_layers=16,
    min_safe_depth=4,
    supported_early_exit=True,
    supported_mod=True,
    layer_cost_usd=0.08 / 16.0,
    layer_latency_ms=8.0 / 16.0,
    layer_flops=1.0e9 / 16.0,
    exit_points=[6, 10, 14, 16],
    quality_curve=_generate_depth_quality_curve(16, 0.74, 0.50),
    is_synthetic=True,
)

SMALL_1_5B_DEPTH_PROFILE = ModelDepthProfile(
    model_name="aeon-small-1.5b",
    total_layers=28,
    min_safe_depth=4,
    supported_early_exit=True,
    supported_mod=True,
    layer_cost_usd=0.30 / 28.0,
    layer_latency_ms=25.0 / 28.0,
    layer_flops=3.0e9 / 28.0,
    exit_points=[10, 16, 22, 28],
    quality_curve=_generate_depth_quality_curve(28, 0.86, 0.58),
    is_synthetic=True,
)

MEDIUM_7B_DEPTH_PROFILE = ModelDepthProfile(
    model_name="aeon-medium-7b",
    total_layers=32,
    min_safe_depth=8,
    supported_early_exit=True,
    supported_mod=True,
    layer_cost_usd=1.00 / 32.0,
    layer_latency_ms=75.0 / 32.0,
    layer_flops=1.4e10 / 32.0,
    exit_points=[12, 18, 24, 32],
    quality_curve=_generate_depth_quality_curve(32, 0.92, 0.65),
    is_synthetic=True,
)

LARGE_70B_DEPTH_PROFILE = ModelDepthProfile(
    model_name="aeon-large-70b",
    total_layers=80,
    min_safe_depth=8,
    supported_early_exit=True,
    supported_mod=True,
    layer_cost_usd=5.00 / 80.0,
    layer_latency_ms=280.0 / 80.0,
    layer_flops=1.4e11 / 80.0,
    exit_points=[24, 40, 56, 72, 80],
    quality_curve=_generate_depth_quality_curve(80, 0.97, 0.72),
    is_synthetic=True,
)

DEFAULT_DEPTH_PROFILES: Dict[str, ModelDepthProfile] = {
    DECISION_MODEL_DEPTH_PROFILE.model_name: DECISION_MODEL_DEPTH_PROFILE,
    DRAFT_0_5B_DEPTH_PROFILE.model_name: DRAFT_0_5B_DEPTH_PROFILE,
    SMALL_1_5B_DEPTH_PROFILE.model_name: SMALL_1_5B_DEPTH_PROFILE,
    MEDIUM_7B_DEPTH_PROFILE.model_name: MEDIUM_7B_DEPTH_PROFILE,
    LARGE_70B_DEPTH_PROFILE.model_name: LARGE_70B_DEPTH_PROFILE,
}


class ModelDepthRegistry:
    """Registry maintaining depth operational profiles for model architectures."""

    def __init__(self, profiles: Optional[Dict[str, ModelDepthProfile]] = None):
        self._profiles: Dict[str, ModelDepthProfile] = dict(profiles or DEFAULT_DEPTH_PROFILES)

    def register(self, profile: ModelDepthProfile) -> None:
        self._profiles[profile.model_name] = profile

    def get(self, model_name: str) -> Optional[ModelDepthProfile]:
        return self._profiles.get(model_name)

    def list_models(self) -> List[str]:
        return list(self._profiles.keys())


global_depth_registry = ModelDepthRegistry()


def get_depth_profile(model_name: str) -> Optional[ModelDepthProfile]:
    """Retrieve model depth profile from global registry."""
    return global_depth_registry.get(model_name)


def get_model_depth_profile(model_name: Optional[str] = None) -> ModelDepthProfile:
    """Retrieve model depth profile, falling back to small profile if None or unknown."""
    if not model_name:
        return SMALL_1_5B_DEPTH_PROFILE
    prof = global_depth_registry.get(model_name)
    return prof if prof is not None else SMALL_1_5B_DEPTH_PROFILE


