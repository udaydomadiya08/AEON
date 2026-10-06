"""
AEON Phase 10 — Routes Package.
"""

from aeon.server.routes.chat import router as chat_router
from aeon.server.routes.models import router as models_router
from aeon.server.routes.health import router as health_router
from aeon.server.routes.metrics import router as metrics_router
from aeon.server.routes.admin import router as admin_router

__all__ = [
    "chat_router",
    "models_router",
    "health_router",
    "metrics_router",
    "admin_router",
]
