"""
AEON Phase 10 — Health & Version Routes.

Exposes standard Kubernetes/cloud liveness, readiness, and build metadata endpoints.
"""

from fastapi import APIRouter, Response, status
from aeon.server.schemas import HealthResponse, VersionResponse
from aeon.server.dependencies import get_health_manager

router = APIRouter(tags=["Health & Diagnostics"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health overview",
    description="Check comprehensive component readiness and system status."
)
async def get_health(response: Response) -> HealthResponse:
    health_mgr = get_health_manager()
    report = health_mgr.check_readiness()
    if report.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Returns 200 OK if the application server process is running."
)
async def get_liveness():
    health_mgr = get_health_manager()
    return health_mgr.check_liveness()


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description="Returns 200 OK only when the orchestrator engine is ready to service inference."
)
async def get_readiness(response: Response) -> HealthResponse:
    health_mgr = get_health_manager()
    report = health_mgr.check_readiness()
    if report.status != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Version metadata",
    description="Export software versions, git commit, and execution environment."
)
async def get_version() -> VersionResponse:
    health_mgr = get_health_manager()
    return health_mgr.get_version_info()
