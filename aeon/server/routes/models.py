"""
AEON Phase 10 — Models Endpoint.

Implements GET /v1/models and GET /v1/models/{model_id} exposing genuine model profiles
and their operational capabilities without advertising nonexistent external infrastructure.
"""

import time
from fastapi import APIRouter, Depends, Request
from aeon.server.schemas import ModelListResponse, ModelObject
from aeon.server.dependencies import get_aeon, require_auth
from aeon.server.errors import ResourceNotFoundError

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get(
    "/models",
    response_model=ModelListResponse,
    dependencies=[Depends(require_auth)],
    summary="List available models",
    description="Lists available AEON orchestrator models and local model profiles."
)
async def list_models(raw_request: Request = None) -> ModelListResponse:
    from aeon.server.dependencies import get_server_state
    aeon = get_server_state(raw_request).aeon
    models = []

    # 1. Primary AEON dynamic orchestrator
    models.append(
        ModelObject(
            id="aeon-auto",
            created=1700000000,
            owned_by="aeon",
            capabilities={
                "adaptive_routing": True,
                "dynamic_depth": True,
                "multi_tier_memory": True,
                "tool_routing": True,
                "speculative_decoding": True,
                "hardware_aware": True,
                "supports_streaming": True,
            },
            x_aeon_metadata={
                "description": "Adaptive Economically-Optimized Network dynamic orchestrator",
                "default": True,
                "execution_mode": "hybrid"
            }
        )
    )

    # 2. Add profiles from model registry
    if hasattr(aeon, "registry") and aeon.registry:
        for profile in aeon.registry.list_models():
            p_name = getattr(profile, "name", None) or (profile.model_id.value if hasattr(profile.model_id, "value") else str(profile.model_id))
            models.append(
                ModelObject(
                    id=p_name,
                    created=1700000000,
                    owned_by="aeon",
                    capabilities={
                        "parameter_count_b": profile.param_count_b,
                        "context_window": getattr(profile, "context_window", 8192),
                        "quality_score": profile.quality_score,
                        "supports_streaming": True,
                        "supported_precisions": ["fp32", "fp16", "int8", "int4"],
                        "supports_tools": True,
                        "supports_reasoning": True,
                    },
                    x_aeon_metadata={
                        "base_latency_ms": profile.base_latency_ms,
                        "tokens_per_second": profile.tokens_per_second,
                        "cost_per_1m_input_usd": profile.cost_per_1m_input_usd,
                        "cost_per_1m_output_usd": profile.cost_per_1m_output_usd,
                        "min_memory_mb": profile.min_memory_mb
                    }
                )
            )

    # 3. Canonical aliases for convenient model referencing
    alias_map = {
        "aeon-small": "aeon-small-1.5b",
        "aeon-medium": "aeon-medium-7b",
        "aeon-large": "aeon-large-70b",
    }
    for alias, target in alias_map.items():
        if any(m.id == target for m in models) and not any(m.id == alias for m in models):
            tgt_obj = next(m for m in models if m.id == target)
            models.append(
                ModelObject(
                    id=alias,
                    created=tgt_obj.created,
                    owned_by="aeon",
                    capabilities=dict(tgt_obj.capabilities),
                    x_aeon_metadata={"alias_for": target, **dict(tgt_obj.x_aeon_metadata or {})}
                )
            )

    return ModelListResponse(data=models)


@router.get(
    "/models/{model_id}",
    response_model=ModelObject,
    dependencies=[Depends(require_auth)],
    summary="Retrieve a model",
    description="Retrieve capability metadata for a specific model profile."
)
async def get_model(model_id: str, raw_request: Request = None) -> ModelObject:
    res = await list_models(raw_request)
    for m in res.data:
        if m.id == model_id:
            return m
    raise ResourceNotFoundError(f"Model '{model_id}' does not exist or is not registered.", param="model_id")
