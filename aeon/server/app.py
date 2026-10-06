"""
AEON Phase 10 — FastAPI Application Factory.

Constructs and configures the complete OpenAPI-compliant AEON HTTP orchestrator server.
"""

import os
from typing import Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from aeon.server.config import ServerConfig
from aeon.server.lifecycle import lifespan
from aeon.server.errors import AeonServerError, InternalServerError
from aeon.server.middleware import (
    RequestTrackingMiddleware,
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
)
from aeon.server.routes import (
    chat_router,
    models_router,
    health_router,
    metrics_router,
    admin_router,
)
from aeon.server.dashboard import dashboard_router


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Instantiate and configure the AEON FastAPI orchestration service."""
    cfg = config or ServerConfig()

    app = FastAPI(
        title="AEON Adaptive Inference Orchestration API",
        version="0.10.0",
        description=(
            "Local-first, provider-agnostic adaptive compute orchestration service. "
            "Dynamically routes tasks across models, tools, memory, reasoning, speculation, "
            "execution depth, precision, and hardware accelerators."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    from aeon.server.dependencies import init_server_state
    state = init_server_state(cfg)
    app.state.config = cfg
    app.state.server_state = state

    # 1. Custom Exception Handlers (Prevent traceback leakage)
    @app.exception_handler(AeonServerError)
    async def aeon_server_error_handler(request: Request, exc: AeonServerError):
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first_err = exc.errors()[0] if exc.errors() else {}
        loc = " -> ".join(str(l) for l in first_err.get("loc", []))
        msg = f"Validation failed at '{loc}': {first_err.get('msg', 'invalid parameter')}"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "message": msg,
                    "type": "invalid_request_error",
                    "param": str(first_err.get("loc", [""])[-1]),
                    "code": "validation_error",
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Fallback 500 without leaking python tracebacks
        err = InternalServerError("An unexpected internal error occurred.")
        return err.to_response()

    # 2. Defensive Middlewares
    app.add_middleware(RequestTrackingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, config=cfg)
    app.add_middleware(RequestSizeLimitMiddleware, config=cfg)

    # 3. CORS Configuration (Restricted by default)
    if cfg.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    # 4. Mount Dashboard Static Assets
    static_dir = os.path.join(os.path.dirname(__file__), "dashboard", "static")
    if os.path.isdir(static_dir):
        app.mount("/dashboard/static", StaticFiles(directory=static_dir), name="static")

    # 5. Register Routes
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(admin_router)

    if cfg.dashboard_enabled:
        app.include_router(dashboard_router)

    return app
