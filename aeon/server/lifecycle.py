"""
AEON Phase 10 — Server Lifecycle Management.

Orchestrates startup initialization, startup banners (with secret protection),
public network interface warnings, and graceful termination handlers.
"""

import sys
import logging
import platform
from contextlib import asynccontextmanager
from fastapi import FastAPI
from aeon.server.config import ServerConfig
from aeon.server.dependencies import init_server_state, get_server_state

logger = logging.getLogger("aeon.server")


def print_startup_banner(config: ServerConfig) -> None:
    """Print clean startup banner without exposing credentials."""
    import torch
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    accel = "Apple Silicon MPS" if mps else ("NVIDIA CUDA" if cuda else "CPU Only")

    auth_str = "Enabled (Bearer/X-API-Key)" if config.api_key else "Disabled (Local Dev Mode)"
    dashboard_str = f"http://{config.host}:{config.port}/dashboard" if config.dashboard_enabled else "Disabled"

    banner = f"""
======================================================================
     AEON — Adaptive Economically-Optimized Network Server
======================================================================
Version:      0.10.0
API Base:     http://{config.host}:{config.port}/v1
Host/Port:    {config.host}:{config.port} (Workers: {config.workers})
Hardware:     {accel} ({platform.processor() or platform.machine()})
Execution:    HYBRID (Simulated kernel + Measured controller)
Auth:         {auth_str}
Dashboard:    {dashboard_str}
Docs:         http://{config.host}:{config.port}/docs
======================================================================
"""
    sys.stdout.write(banner + "\n")
    sys.stdout.flush()

    warning = config.get_public_binding_warning()
    if warning:
        sys.stderr.write(f"\n{warning}\n\n")
        sys.stderr.flush()


async def on_shutdown(app: FastAPI) -> None:
    """Explicit graceful shutdown handler for process cleanup."""
    state = get_server_state()
    logger.info("Initiating graceful shutdown for AEON server...")
    if hasattr(state.aeon, "memory_hierarchy") and state.aeon.memory_hierarchy:
        try:
            state.aeon.memory_hierarchy.clear()
        except Exception:
            pass
    logger.info("AEON server shut down successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI async context manager controlling startup and graceful shutdown."""
    config: ServerConfig = getattr(app.state, "config", ServerConfig())

    # Startup
    init_server_state(config)
    print_startup_banner(config)

    yield

    # Graceful Shutdown
    await on_shutdown(app)
