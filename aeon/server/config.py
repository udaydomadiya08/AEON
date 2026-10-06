"""
AEON Phase 10 — Server Configuration.

Provides structured, validated, environment-aware configuration for the
local-first AEON OpenAPI server. Follows strict priority order:
CLI overrides > Environment variables > Configuration file > Safe defaults.
"""

import os
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ServerConfig(BaseModel):
    """Configuration settings for the AEON HTTP orchestration service."""

    # Network binding
    host: str = Field(default="127.0.0.1", description="Server host address (defaults to localhost)")
    port: int = Field(default=8000, description="Server port number")
    workers: int = Field(default=1, description="Number of worker processes")

    # Security & Authentication
    api_key: Optional[str] = Field(default=None, description="Optional bearer token for authenticated requests")
    cors_allowed_origins: List[str] = Field(default_factory=list, description="Allowed CORS origins (restricted by default)")
    security_headers_enabled: bool = Field(default=True, description="Enable secure HTTP response headers")

    # Rate Limiting & Concurrency
    rate_limit_enabled: bool = Field(default=False, description="Enable per-client rate limiting")
    rate_limit_per_minute: int = Field(default=120, description="Maximum requests per minute per client")
    max_concurrency: int = Field(default=16, description="Maximum concurrent active requests")
    queue_size: int = Field(default=64, description="Maximum backlog queue size for backpressure")
    queue_timeout_seconds: float = Field(default=10.0, description="Queue wait timeout before returning 503")

    # Execution Timeouts
    request_timeout_seconds: float = Field(default=30.0, description="End-to-end request timeout")
    backend_timeout_seconds: float = Field(default=25.0, description="Backend model generation timeout")

    # Request Size Limits
    max_request_bytes: int = Field(default=1_048_576, description="Max allowed request payload size (1 MB default)")
    max_messages: int = Field(default=100, description="Maximum message count per chat completion request")
    max_max_tokens: int = Field(default=4096, description="Maximum allowed max_tokens requested by client")

    # Observability & Dashboard
    telemetry_enabled: bool = Field(default=True, description="Collect request execution telemetry")
    telemetry_max_records: int = Field(default=10000, description="Bounded in-memory telemetry buffer limit")
    telemetry_max_age_seconds: int = Field(default=86400, description="Maximum age of telemetry records in seconds (24h)")
    dashboard_enabled: bool = Field(default=True, description="Enable /dashboard web interface")
    log_level: str = Field(default="INFO", description="Log verbosity level (DEBUG, INFO, WARNING, ERROR)")

    # Privacy
    content_logging_enabled: bool = Field(default=False, description="Explicit opt-in to log prompts/completions (default False)")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("max_concurrency")
    @classmethod
    def validate_concurrency(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_concurrency must be at least 1, got {v}")
        return v

    def is_public_binding(self) -> bool:
        """Return True if binding to any public network interface."""
        return self.host in ["0.0.0.0", "::"]

    def get_public_binding_warning(self) -> Optional[str]:
        """Return a security warning if binding to a non-localhost address without authentication."""
        if self.is_public_binding():
            msg = "WARNING: AEON server is bound to a public interface (" + self.host + ")."
            if not self.api_key:
                msg += " Authentication is DISABLED. It is strongly advised to configure AEON_API_KEY."
            return msg
        return None

    def redact_for_display(self) -> Dict[str, Any]:
        """Export config dictionary with sensitive keys redacted."""
        data = self.model_dump()
        if data.get("api_key"):
            data["api_key"] = "[REDACTED]"
        return data

    @classmethod
    def from_env(cls, config_file: Optional[str] = None, **cli_overrides) -> "ServerConfig":
        """
        Instantiate ServerConfig with strict precedence:
        CLI overrides > Environment variables > Config file > Defaults.
        """
        file_values: Dict[str, Any] = {}
        if config_file and os.path.isfile(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    file_values = json.load(f)
            except Exception as e:
                pass

        env_values: Dict[str, Any] = {}
        if "AEON_HOST" in os.environ:
            env_values["host"] = os.environ["AEON_HOST"]
        if "AEON_PORT" in os.environ:
            try:
                env_values["port"] = int(os.environ["AEON_PORT"])
            except ValueError:
                pass
        if "AEON_WORKERS" in os.environ:
            try:
                env_values["workers"] = int(os.environ["AEON_WORKERS"])
            except ValueError:
                pass
        if "AEON_API_KEY" in os.environ:
            env_values["api_key"] = os.environ["AEON_API_KEY"]
        if "AEON_CORS_ORIGINS" in os.environ:
            origins = [o.strip() for o in os.environ["AEON_CORS_ORIGINS"].split(",") if o.strip()]
            env_values["cors_allowed_origins"] = origins
        if "AEON_RATE_LIMIT_ENABLED" in os.environ:
            env_values["rate_limit_enabled"] = os.environ["AEON_RATE_LIMIT_ENABLED"].lower() in ["1", "true", "yes"]
        if "AEON_RATE_LIMIT_PER_MINUTE" in os.environ:
            try:
                env_values["rate_limit_per_minute"] = int(os.environ["AEON_RATE_LIMIT_PER_MINUTE"])
            except ValueError:
                pass
        if "AEON_MAX_CONCURRENCY" in os.environ:
            try:
                env_values["max_concurrency"] = int(os.environ["AEON_MAX_CONCURRENCY"])
            except ValueError:
                pass
        if "AEON_DASHBOARD_ENABLED" in os.environ:
            env_values["dashboard_enabled"] = os.environ["AEON_DASHBOARD_ENABLED"].lower() in ["1", "true", "yes"]
        if "AEON_LOG_LEVEL" in os.environ:
            env_values["log_level"] = os.environ["AEON_LOG_LEVEL"].upper()
        if "AEON_CONTENT_LOGGING" in os.environ:
            env_values["content_logging_enabled"] = os.environ["AEON_CONTENT_LOGGING"].lower() in ["1", "true", "yes"]

        # Merge with precedence: file_values < env_values < cli_overrides
        merged: Dict[str, Any] = {}
        merged.update(file_values)
        merged.update(env_values)
        # Filter out None from cli_overrides so defaults aren't masked
        filtered_cli = {k: v for k, v in cli_overrides.items() if v is not None}
        merged.update(filtered_cli)

        return cls(**merged)
