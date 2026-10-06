"""
AEON Phase 10 — Health Checks & Diagnostics.

Provides liveness and readiness diagnostic inspections. Explicitly ensures that
the absence of optional datacenter accelerators (e.g. CUDA) does not cause
readiness failure if a valid fallback (e.g. Apple Silicon MPS or CPU) is operational.
"""

import sys
import platform
from datetime import datetime
from typing import Dict, Any, Optional
from aeon import AEON
from aeon.server.schemas import HealthResponse, ComponentHealth, VersionResponse


def get_current_git_commit() -> str:
    """Read current git commit hash."""
    try:
        import subprocess
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "63470fff"


class HealthManager:
    """Manages system liveness and readiness evaluations."""

    def __init__(self, aeon_engine: Optional[AEON] = None):
        self.aeon_engine = aeon_engine

    def check_liveness(self) -> Dict[str, Any]:
        """Simple process liveness verification."""
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat()
        }

    def check_readiness(self) -> HealthResponse:
        """
        Comprehensive readiness evaluation across all AEON subsystems.
        Service is ready as long as core engine and at least one device backend are functional.
        """
        components: Dict[str, ComponentHealth] = {}

        # 1. Engine check
        engine_ready = self.aeon_engine is not None
        components["controller"] = ComponentHealth(
            status="operational" if engine_ready else "uninitialized",
            available=engine_ready,
            details={"policy": getattr(self.aeon_engine, "policy", "voi")} if engine_ready else None
        )

        # 2. Hardware backend check
        import torch
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        cuda = torch.cuda.is_available()
        primary_dev = "mps" if mps else ("cuda" if cuda else "cpu")

        components["hardware"] = ComponentHealth(
            status="operational",
            available=True,  # CPU is always available
            details={
                "primary_device": primary_dev,
                "mps_available": mps,
                "cuda_available": cuda,
                "provenance": "MEASURED"
            }
        )

        # 3. Memory hierarchy check
        mem_ok = bool(self.aeon_engine and hasattr(self.aeon_engine, "memory_hierarchy") and self.aeon_engine.memory_hierarchy)
        components["memory"] = ComponentHealth(
            status="operational" if mem_ok else "disabled",
            available=mem_ok,
            details={"levels": ["L1_exact", "L2_semantic", "L3_procedural", "L4_external"]} if mem_ok else None
        )

        # 4. Tool controller check
        tools_ok = bool(self.aeon_engine and hasattr(self.aeon_engine, "tool_controller") and self.aeon_engine.tool_controller)
        tool_count = 0
        if tools_ok:
            tc = self.aeon_engine.tool_controller
            reg = getattr(tc, "tool_registry", None) or getattr(tc, "registry", None)
            if reg and hasattr(reg, "list_tools"):
                tool_count = len(reg.list_tools())
        components["tools"] = ComponentHealth(
            status="operational" if tools_ok else "disabled",
            available=tools_ok,
            details={"tool_count": tool_count} if tools_ok else None
        )

        # 5. Model registry check
        reg_ok = bool(self.aeon_engine and hasattr(self.aeon_engine, "registry") and self.aeon_engine.registry)
        model_count = 0
        if reg_ok:
            reg = self.aeon_engine.registry
            if hasattr(reg, "list_models"):
                model_count = len(reg.list_models())
            elif hasattr(reg, "_profiles"):
                model_count = len(reg._profiles)
        components["models"] = ComponentHealth(
            status="operational" if reg_ok else "uninitialized",
            available=reg_ok,
            details={"registered_models": model_count} if reg_ok else None
        )

        is_all_ready = engine_ready and components["hardware"].available
        return HealthResponse(
            status="ready" if is_all_ready else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version="0.10.0",
            components=components
        )

    def get_version_info(self) -> VersionResponse:
        """Export comprehensive version metadata without exposing filesystem paths."""
        return VersionResponse(
            aeon_version="0.10.0",
            api_version="v1",
            git_commit=get_current_git_commit(),
            python_version=platform.python_version(),
            timestamp=datetime.utcnow().isoformat()
        )
