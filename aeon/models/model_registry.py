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


# Global default registry instance
global_registry = ModelRegistry()
