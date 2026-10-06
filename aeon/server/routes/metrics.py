"""
AEON Phase 10 — Metrics Exporter Route.

Exposes machine-readable operational metrics supporting both JSON API consumers
and Prometheus metrics scrapers via GET /metrics.
"""

from typing import Optional
from fastapi import APIRouter, Request, Response
from starlette.responses import PlainTextResponse
from aeon.server.dependencies import get_telemetry_aggregator
from aeon.server.telemetry.aggregator import format_prometheus_metrics

router = APIRouter(tags=["Observability"])


@router.get(
    "/metrics",
    summary="Export telemetry metrics",
    description="Returns analytical metrics in Prometheus text exposition or JSON format."
)
async def get_metrics(request: Request, format: Optional[str] = None):
    aggregator = get_telemetry_aggregator()
    summary = await aggregator.get_summary()

    # Determine requested content type
    accept = request.headers.get("accept", "")
    wants_json = format == "json" or "application/json" in accept

    if wants_json:
        return summary

    prom_text = format_prometheus_metrics(summary)
    return PlainTextResponse(content=prom_text, media_type="text/plain; version=0.0.4; charset=utf-8")
