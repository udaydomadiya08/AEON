"""
AEON Phase 10 — Authentication & Secret Protection.

Implements optional API key verification using constant-time comparison.
Includes utilities for redacting sensitive secrets and authorization headers from logs.
"""

import re
import hmac
from typing import Optional
from starlette.requests import Request
from aeon.server.config import ServerConfig
from aeon.server.errors import AuthenticationError


def verify_api_key(request: Request, config: ServerConfig) -> None:
    """
    Validate incoming client API key against server configuration.
    If server config has no api_key set, authentication is bypassed (local dev default).
    If api_key is configured, requests must supply matching token via Authorization or X-API-Key header.
    """
    if not config.api_key:
        return  # Auth disabled for local development

    client_key: Optional[str] = None

    # Check Authorization header (Bearer token)
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            client_key = parts[1]

    # Fallback to X-API-Key header
    if not client_key:
        client_key = request.headers.get("X-API-Key")

    if not client_key:
        raise AuthenticationError("Missing API key. Provide Bearer token or X-API-Key header.")

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(client_key.encode("utf-8"), config.api_key.encode("utf-8")):
        raise AuthenticationError("Invalid API key provided.")


def constant_time_compare(val1: Optional[str], val2: Optional[str]) -> bool:
    """Perform constant-time comparison between two strings to mitigate timing side-channels."""
    if val1 is None or val2 is None:
        return False
    return hmac.compare_digest(str(val1).encode("utf-8"), str(val2).encode("utf-8"))


SECRET_PATTERNS = [
    (re.compile(r"Bearer\s+([A-Za-z0-9_\-\.]{6,})", re.IGNORECASE), r"Bearer [REDACTED]"),
    (re.compile(r"X-API-Key:\s*([A-Za-z0-9_\-\.]{6,})", re.IGNORECASE), r"X-API-Key: [REDACTED]"),
    (re.compile(r"api[_\-]?key(?:\s+is\s+|['\":\s=]+)([A-Za-z0-9_\-\.]{6,})", re.IGNORECASE), r"api_key: [REDACTED]"),
    (re.compile(r"password(?:\s+is\s+|['\":\s=]+)([^'\"\s,;]+)", re.IGNORECASE), r"password: [REDACTED]"),
    (re.compile(r"secret(?:\s+is\s+|['\":\s=]+)([^'\"\s,;]+)", re.IGNORECASE), r"secret: [REDACTED]"),
]


def redact_secrets(text: str) -> str:
    """Mask known sensitive patterns from logs and diagnostic traces."""
    if not text:
        return text
    sanitized = text
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized
