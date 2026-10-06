"""
AEON Phase 10 — Dashboard Routes.

Serves the lightweight, zero-dependency HTML/JS local dashboard and analytical summary API.
"""

import os
import time
import platform
from typing import Dict, Any, List
from fastapi import APIRouter
from starlette.responses import HTMLResponse, JSONResponse
from aeon.server.dependencies import get_server_state
from aeon.server.dashboard.schemas import SystemInfo, DashboardSummary

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

SERVER_START_TIME = time.time()


@router.get("", response_class=HTMLResponse, summary="Dashboard UI")
@router.get("/", response_class=HTMLResponse, summary="Dashboard UI")
async def get_dashboard_html():
    """Serve the static single-page dashboard application."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    html_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>AEON Dashboard</h1><p>Static bundle not found.</p>")


@router.get("/api/summary", response_model=DashboardSummary, summary="Dashboard metrics summary")
async def get_dashboard_summary():
    """Export aggregated operational, routing, and economic metrics for dashboard rendering."""
    state = get_server_state()
    aggregator = state.telemetry_aggregator
    metrics_data = await aggregator.get_summary()

    import torch
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    device = "mps" if mps else ("cuda" if cuda else "cpu")

    sys_info = SystemInfo(
        cpu=platform.processor() or platform.machine(),
        ram_gb=16.0,
        device=device,
        mps_available=mps,
        cuda_available=cuda,
        python_version=platform.python_version(),
        aeon_version="0.10.0",
        uptime_seconds=round(time.time() - SERVER_START_TIME, 1)
    )

    return DashboardSummary(
        system=sys_info,
        metrics=metrics_data,
        provenance=metrics_data.get("provenance", {})
    )


@router.get("/api/recent", summary="Recent request records")
async def get_recent_requests(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve recent sanitized request metadata."""
    state = get_server_state()
    return await state.telemetry_store.get_recent_records(limit=min(limit, 100))
