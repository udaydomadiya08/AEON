"""
AEON Phase 10 — Dashboard Schemas.

Defines analytical summary models for the web dashboard.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class SystemInfo(BaseModel):
    cpu: str
    ram_gb: float
    device: str
    mps_available: bool
    cuda_available: bool
    python_version: str
    aeon_version: str
    uptime_seconds: float


class DashboardSummary(BaseModel):
    system: SystemInfo
    metrics: Dict[str, Any]
    provenance: Dict[str, str]
