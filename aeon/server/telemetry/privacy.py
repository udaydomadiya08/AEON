"""
AEON Phase 10 — Telemetry Privacy & PII Redaction.

Protects user privacy by stripping prompt/completion content by default and
redacting sensitive credentials, tokens, and PII from telemetry events.
"""

import re
from typing import Dict, Any, Optional

SENSITIVE_KEY_PATTERNS = [
    re.compile(r"api[_\-]?key", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"bearer", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"(?:user|auth|access|bearer|session|csrf)[_\-]?token", re.IGNORECASE),
]

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """Sanitize potential PII / secret values from textual payload."""
    if not text:
        return text
    from aeon.server.auth import redact_secrets
    redacted = redact_secrets(text)
    redacted = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", redacted)
    redacted = PHONE_PATTERN.sub("[PHONE_REDACTED]", redacted)
    return redacted


def sanitize_telemetry_event(event: Dict[str, Any], allow_content: bool = False) -> Dict[str, Any]:
    """
    Sanitize a telemetry record prior to persistence.
    If allow_content is False, all prompt and completion content fields are suppressed.
    """
    sanitized = dict(event)

    # Privacy by default: purge raw textual content
    if not allow_content:
        sanitized.pop("prompt", None)
        sanitized.pop("response", None)
        sanitized.pop("messages", None)
    else:
        if "prompt" in sanitized and isinstance(sanitized["prompt"], str):
            sanitized["prompt"] = sanitize_text(sanitized["prompt"])
        if "response" in sanitized and isinstance(sanitized["response"], str):
            sanitized["response"] = sanitize_text(sanitized["response"])

    # Scrub sensitive metadata keys
    for k in list(sanitized.keys()):
        for pat in SENSITIVE_KEY_PATTERNS:
            if pat.search(k):
                sanitized[k] = "[REDACTED]"
                break

    return sanitized
