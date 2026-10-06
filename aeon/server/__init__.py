"""
AEON Phase 10 — OpenAPI Server & Dashboard Package.
"""

from aeon.server.config import ServerConfig
from aeon.server.app import create_app

__all__ = ["create_app", "ServerConfig"]
