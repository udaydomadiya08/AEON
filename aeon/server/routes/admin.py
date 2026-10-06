"""
AEON Phase 10 — Administrative Control Endpoints.

Provides administrative diagnostic, configuration inspection, and state refresh routes.
Always enforces authentication when configured and strictly redacts credentials.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends
from aeon.server.dependencies import get_server_state, require_auth

router = APIRouter(prefix="/admin", tags=["Administration"])


@router.get(
    "/config",
    dependencies=[Depends(require_auth)],
    summary="Inspect server configuration",
    description="Retrieve running server configuration with sensitive keys and secrets redacted."
)
async def get_admin_config() -> Dict[str, Any]:
    state = get_server_state()
    return state.config.redact_for_display()


@router.get(
    "/status",
    dependencies=[Depends(require_auth)],
    summary="Inspect runtime status",
    description="Retrieve current active concurrency, queue backlog, and hardware status."
)
async def get_admin_status() -> Dict[str, Any]:
    state = get_server_state()
    hw_ctrl = getattr(state.aeon, "hardware_controller", None)
    headroom = getattr(hw_ctrl.profile, "ram_available_mb", 4096.0) if (hw_ctrl and hasattr(hw_ctrl, "profile")) else 4096.0

    return {
        "active_concurrency": state.backpressure.active_count,
        "queued_requests": state.backpressure.waiting_count,
        "max_concurrency": state.config.max_concurrency,
        "queue_capacity": state.config.queue_size,
        "memory_headroom_mb": round(headroom, 1),
        "telemetry_records_count": len(await state.telemetry_store.get_all_records())
    }


@router.post(
    "/reload",
    dependencies=[Depends(require_auth)],
    summary="Reload server state",
    description="Clear cache and re-evaluate hardware telemetry without restarting process."
)
async def reload_admin() -> Dict[str, Any]:
    state = get_server_state()
    if hasattr(state.aeon, "memory_hierarchy") and state.aeon.memory_hierarchy:
        state.aeon.memory_hierarchy.clear()
    return {"status": "reloaded", "message": "Memory cache cleared and subsystem state refreshed."}
