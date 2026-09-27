"""
Model Registry for AEON.
Manages available models, capabilities, and selection lookups.
"""

from typing import Dict, List, Optional
from aeon.types import ModelProfile, RouteChoice
from aeon.models.model_profiles import (
    DEFAULT_PROFILES,
    DECISION_MODEL_PROFILE,
    SMALL_MODEL_PROFILE,
    MEDIUM_MODEL_PROFILE,
    LARGE_MODEL_PROFILE,
)


class ModelRegistry:
    """Registry maintaining active model profiles and hardware-aware selection."""

    def __init__(self, profiles: Optional[Dict[str, ModelProfile]] = None):
        self._profiles: Dict[str, ModelProfile] = profiles or dict(DEFAULT_PROFILES)

    def register(self, profile: ModelProfile) -> None:
        self._profiles[profile.name] = profile

    def get(self, name: str) -> Optional[ModelProfile]:
        return self._profiles.get(name)

    def list_models(self) -> List[ModelProfile]:
        return list(self._profiles.values())

    def get_model_for_route(self, route: RouteChoice) -> ModelProfile:
        """Return the default registered model profile corresponding to an execution route."""
        if route == RouteChoice.DECISION_MODEL:
            return self._profiles.get(DECISION_MODEL_PROFILE.name, DECISION_MODEL_PROFILE)
        elif route == RouteChoice.SMALL_MODEL:
            return self._profiles.get(SMALL_MODEL_PROFILE.name, SMALL_MODEL_PROFILE)
        elif route == RouteChoice.MEDIUM_MODEL:
            return self._profiles.get(MEDIUM_MODEL_PROFILE.name, MEDIUM_MODEL_PROFILE)
        elif route == RouteChoice.LARGE_MODEL:
            return self._profiles.get(LARGE_MODEL_PROFILE.name, LARGE_MODEL_PROFILE)
        else:
            return self._profiles.get(SMALL_MODEL_PROFILE.name, SMALL_MODEL_PROFILE)

    def list_draft_models(self, target_model: Optional[str] = None) -> List[ModelProfile]:
        """List all registered draft or dual-role models, optionally filtered by target compatibility."""
        drafts = [m for m in self._profiles.values() if m.role in ["DRAFT", "DUAL"]]
        if target_model:
            return [m for m in drafts if self.is_compatible(m.name, target_model)]
        return drafts

    def is_compatible(self, draft_model_name: str, target_model_name: str) -> bool:
        """Verify whether draft_model is architecturally and parameter-compatible with target_model."""
        draft = self._profiles.get(draft_model_name)
        target = self._profiles.get(target_model_name)
        if not draft or not target:
            return False
        if draft.name == target.name:
            return False
        # Draft model must have strictly fewer parameters than target model
        if draft.param_count_b >= target.param_count_b:
            return False
        # Explicit compatibility list check
        if draft.compatible_targets and target.name not in draft.compatible_targets:
            return False
        return True

    def get_best_draft_model(self, target_model_name: str) -> Optional[ModelProfile]:
        """Select the most economical compatible draft model for a target model."""
        candidates = self.list_draft_models(target_model_name)
        if not candidates:
            return None
        # Sort by parameter count / latency to pick the most efficient draft
        return sorted(candidates, key=lambda m: (m.param_count_b, m.base_latency_ms))[0]


# Global default registry instance
global_registry = ModelRegistry()
