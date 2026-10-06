"""
AEON Phase 10 — FastAPI Dependency Injection Container.

Provides clean, single-point access to the AEON engine, configuration,
adapters, telemetry stores, and security verifiers for route handlers.
"""

from typing import Optional
from starlette.requests import Request
from aeon import AEON
from aeon.server.config import ServerConfig
from aeon.server.auth import verify_api_key
from aeon.server.adapters.openai_compatible import OpenAICompatibleAdapter
from aeon.server.telemetry.store import TelemetryStore
from aeon.server.telemetry.aggregator import TelemetryAggregator
from aeon.server.health import HealthManager
from aeon.server.middleware import InMemoryRateLimiter, ConcurrencyBackpressureManager


class ServerState:
    """Singleton-style state holder for the AEON HTTP service instance."""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.aeon = AEON(
            policy="voi",
            memory=True,
            tools=True,
            speculation=True,
            reasoning=True,
            adaptive_depth=True,
            hardware_aware=True,
        )
        self.adapter = OpenAICompatibleAdapter(self.aeon)
        self.telemetry_store = TelemetryStore(
            max_records=self.config.telemetry_max_records,
            max_age_seconds=self.config.telemetry_max_age_seconds,
            allow_content_logging=self.config.content_logging_enabled,
        )
        self.telemetry_aggregator = TelemetryAggregator(self.telemetry_store)
        self.health_manager = HealthManager(self.aeon)
        self.rate_limiter = InMemoryRateLimiter(max_per_minute=self.config.rate_limit_per_minute)
        self.backpressure = ConcurrencyBackpressureManager(
            max_concurrency=self.config.max_concurrency,
            queue_size=self.config.queue_size,
            queue_timeout=self.config.queue_timeout_seconds,
        )


_global_server_state: Optional[ServerState] = None


def init_server_state(config: Optional[ServerConfig] = None) -> ServerState:
    """Initialize or update global server state."""
    global _global_server_state
    _global_server_state = ServerState(config)
    return _global_server_state


def get_server_state(request: Optional[Request] = None) -> ServerState:
    """Access current active server state, prioritizing the requesting app's state."""
    if request is not None and hasattr(request, "app") and hasattr(request.app.state, "server_state"):
        return request.app.state.server_state
    global _global_server_state
    if _global_server_state is None:
        _global_server_state = ServerState()
    return _global_server_state


def get_config(request: Optional[Request] = None) -> ServerConfig:
    return get_server_state(request).config


def get_aeon(request: Optional[Request] = None) -> AEON:
    return get_server_state(request).aeon


def get_adapter(request: Optional[Request] = None) -> OpenAICompatibleAdapter:
    return get_server_state(request).adapter


def get_telemetry_store(request: Optional[Request] = None) -> TelemetryStore:
    return get_server_state(request).telemetry_store


def get_telemetry_aggregator(request: Optional[Request] = None) -> TelemetryAggregator:
    return get_server_state(request).telemetry_aggregator


def get_health_manager(request: Optional[Request] = None) -> HealthManager:
    return get_server_state(request).health_manager


async def require_auth(request: Request) -> None:
    """Dependency enforcing API key authentication if enabled."""
    state = get_server_state(request)
    verify_api_key(request, state.config)


async def check_rate_limits(request: Request) -> None:
    """Dependency applying per-client rate limit checks."""
    state = get_server_state(request)
    if state.config.rate_limit_enabled:
        client_ip = request.client.host if request.client else "unknown"
        await state.rate_limiter.check_rate_limit(client_ip)
