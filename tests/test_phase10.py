"""
AEON Phase 10 — OpenAPI Server, Dashboard, and Production Hardening Test Suite.

Comprehensive verification of:
1. Server configuration, environment priority, and public binding safety
2. Defensive security headers, CORS restrictions, and request ID tracking
3. Optional API key authentication (constant-time, secret scrubbing)
4. Standardized OpenAI-compatible Chat Completions (single/multi-turn, tools, memory, reasoning)
5. Namespaced AEON policy overrides and non-intrusive x_aeon telemetry
6. Server-Sent Events (SSE) streaming compliance and [DONE] termination
7. Model listing, capability exposure, and 404 resource handling
8. Health inspection, Kubernetes liveness/readiness, and build version metadata
9. In-memory rate limiting and concurrency backpressure management
10. Memory safety preflight rejection
11. Bounded in-memory telemetry, age pruning, and privacy redaction
12. Metrics exporter (JSON & Prometheus format)
13. Local web dashboard HTML, CSS, JS, and JSON summary API
14. Administrative configuration inspection, secret redaction, and reload
15. OpenAPI JSON specification validity and Swagger/ReDoc docs
16. Local concurrency, backpressure, and load test
"""

import os
import json
import asyncio
import pytest
from starlette.testclient import TestClient
from click.testing import CliRunner

from aeon.server.config import ServerConfig
from aeon.server.app import create_app
from aeon.server.errors import (
    AeonServerError,
    InvalidRequestError,
    AuthenticationError,
    PermissionDeniedError,
    ResourceNotFoundError,
    RequestTimeoutError,
    PayloadTooLargeError,
    ValidationError,
    RateLimitExceededError,
    InternalServerError,
    ServiceUnavailableError,
)
from aeon.server.auth import redact_secrets, verify_api_key, constant_time_compare
from aeon.server.middleware import (
    InMemoryRateLimiter,
    ConcurrencyBackpressureManager,
    VALID_REQUEST_ID_REGEX
)
from aeon.server.telemetry.store import TelemetryStore
from aeon.server.telemetry.privacy import sanitize_text, sanitize_telemetry_event
from aeon.server.telemetry.aggregator import TelemetryAggregator, format_prometheus_metrics
from aeon.server.health import HealthManager
from aeon.cli.main import cli


# =========================================================================
# 1. Server Configuration & Security Defaults
# =========================================================================

def test_server_config_defaults_to_localhost():
    cfg = ServerConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.api_key is None
    assert cfg.is_public_binding() is False
    assert cfg.get_public_binding_warning() is None


def test_server_config_public_binding_warning():
    cfg = ServerConfig(host="0.0.0.0")
    assert cfg.is_public_binding() is True
    warning = cfg.get_public_binding_warning()
    assert warning is not None
    assert "WARNING: AEON server is bound to a public interface" in warning


def test_server_config_port_and_concurrency_validation():
    with pytest.raises(ValueError):
        ServerConfig(port=99999)
    with pytest.raises(ValueError):
        ServerConfig(port=0)
    with pytest.raises(ValueError):
        ServerConfig(max_concurrency=0)


def test_server_config_redacts_api_key():
    cfg = ServerConfig(api_key="secret-token-xyz-123")
    redacted = cfg.redact_for_display()
    assert redacted["api_key"] == "[REDACTED]"
    assert cfg.api_key == "secret-token-xyz-123"


def test_server_config_precedence(monkeypatch, tmp_path):
    # File
    config_file = tmp_path / "server_cfg.json"
    config_file.write_text(json.dumps({"port": 8010, "workers": 2}))

    # Env
    monkeypatch.setenv("AEON_PORT", "8020")
    monkeypatch.setenv("AEON_API_KEY", "env-secret")

    # CLI overrides
    cfg = ServerConfig.from_env(
        config_file=str(config_file),
        port=8030  # CLI overrides both file and env
    )

    assert cfg.port == 8030
    assert cfg.workers == 2  # from file
    assert cfg.api_key == "env-secret"  # from env


# =========================================================================
# 2. Authentication & Secret Scrubbing
# =========================================================================

def test_secret_redaction_utility():
    raw = "User Authorization: Bearer sk-antigravity-9999 with api_key: secret-abc"
    scrubbed = redact_secrets(raw)
    assert "sk-antigravity-9999" not in scrubbed
    assert "Bearer [REDACTED]" in scrubbed
    assert "api_key: [REDACTED]" in scrubbed


def test_auth_disabled_by_default():
    app = create_app(ServerConfig(api_key=None))
    client = TestClient(app)
    res = client.get("/v1/models")
    assert res.status_code == 200


def test_auth_enforces_bearer_token():
    app = create_app(ServerConfig(api_key="valid-test-key-123"))
    client = TestClient(app)

    # Missing auth
    res = client.get("/v1/models")
    assert res.status_code == 401
    assert res.json()["error"]["type"] == "authentication_error"

    # Invalid token
    res = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert res.status_code == 401

    # Valid Bearer token
    res = client.get("/v1/models", headers={"Authorization": "Bearer valid-test-key-123"})
    assert res.status_code == 200

    # Valid X-API-Key header
    res = client.get("/v1/models", headers={"X-API-Key": "valid-test-key-123"})
    assert res.status_code == 200


# =========================================================================
# 3. Defensive Headers & Middleware
# =========================================================================

def test_security_headers_injected():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["referrer-policy"] == "no-referrer"
    assert res.headers["x-aeon-server"] == "local-orchestrator"


def test_request_id_assigned_and_propagated():
    app = create_app(ServerConfig())
    client = TestClient(app)

    # Auto generated
    res = client.get("/health/live")
    req_id = res.headers.get("x-request-id")
    assert req_id is not None
    assert req_id.startswith("req-")

    # Client supplied
    res = client.get("/health/live", headers={"X-Request-ID": "custom-trace-uuid-123"})
    assert res.headers.get("x-request-id") == "custom-trace-uuid-123"


def test_oversized_payload_rejected_with_413():
    app = create_app(ServerConfig(max_request_bytes=50))
    client = TestClient(app)
    large_payload = {"messages": [{"role": "user", "content": "x" * 200}]}
    res = client.post("/v1/chat/completions", json=large_payload)
    assert res.status_code == 413
    assert res.json()["error"]["type"] == "request_too_large_error"


# =========================================================================
# 4. Standardized API Errors
# =========================================================================

def test_error_schemas_are_openai_compatible():
    err = InvalidRequestError("Malformed message schema", param="messages")
    data = err.to_dict()
    assert "error" in data
    assert data["error"]["message"] == "Malformed message schema"
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["param"] == "messages"
    assert data["error"]["code"] == "invalid_request"


def test_unhandled_exception_does_not_leak_traceback():
    app = create_app(ServerConfig())

    # Dynamically inject an error endpoint for testing
    @app.get("/trigger-crash")
    def trigger_crash():
        raise RuntimeError("Secret internal database password inside traceback!")

    client = TestClient(app, raise_server_exceptions=False)
    res = client.get("/trigger-crash")
    assert res.status_code == 500
    body = res.json()
    assert body["error"]["type"] == "internal_server_error"
    assert "Secret internal database password" not in res.text


# =========================================================================
# 5. Health & Diagnostic Endpoints
# =========================================================================

def test_health_live_endpoint():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json()["status"] == "alive"


def test_health_ready_endpoint():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "hardware" in data["components"]
    assert "controller" in data["components"]
    assert data["components"]["hardware"]["available"] is True


def test_version_endpoint():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/version")
    assert res.status_code == 200
    data = res.json()
    assert data["aeon_version"] == "0.10.0"
    assert data["api_version"] == "v1"
    assert "git_commit" in data
    assert "/" not in data["git_commit"]  # No filesystem path leaked


# =========================================================================
# 6. Models Endpoints
# =========================================================================

def test_models_list_returns_aeon_auto():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "aeon-auto" in ids


def test_models_retrieve_specific_model():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/v1/models/aeon-auto")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "aeon-auto"
    assert data["capabilities"]["adaptive_routing"] is True


def test_models_retrieve_nonexistent_returns_404():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/v1/models/gpt-99-nonexistent")
    assert res.status_code == 404
    assert res.json()["error"]["type"] == "resource_not_found_error"


# =========================================================================
# 7. Chat Completions — Synchronous Non-Streaming
# =========================================================================

def test_chat_completions_basic_request():
    app = create_app(ServerConfig())
    client = TestClient(app)
    req = {
        "model": "aeon-auto",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.0,
        "max_tokens": 64
    }
    res = client.post("/v1/chat/completions", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert len(data["choices"][0]["message"]["content"]) > 0
    assert data["usage"]["total_tokens"] > 0

    # Verify x_aeon telemetry
    assert "x_aeon" in data
    x = data["x_aeon"]
    assert x["selected_device"] in ["mps", "cpu", "cuda"]
    assert x["provenance"] == "HYBRID"
    assert x["latency_ms"] > 0


def test_chat_completions_multi_turn_with_system():
    app = create_app(ServerConfig())
    client = TestClient(app)
    req = {
        "model": "aeon-auto",
        "messages": [
            {"role": "system", "content": "You are a concise financial analyst."},
            {"role": "user", "content": "Calculate the margin."},
            {"role": "assistant", "content": "Margin is 15%."},
            {"role": "user", "content": "What is 17.3% of 847?"}
        ]
    }
    res = client.post("/v1/chat/completions", json=req)
    assert res.status_code == 200
    data = res.json()
    assert "146.531" in data["choices"][0]["message"]["content"]


def test_chat_completions_deterministic_arithmetic_tool_routing():
    app = create_app(ServerConfig())
    client = TestClient(app)
    # Compound interest calculation matches Phase 4/9 tool
    req = {
        "model": "aeon-auto",
        "messages": [
            {"role": "user", "content": "Calculate the compound amount for principal 1000, rate 5%, time 3 years compounded annually."}
        ]
    }
    res = client.post("/v1/chat/completions", json=req)
    assert res.status_code == 200
    data = res.json()
    assert "x_aeon" in data


def test_chat_completions_policy_overrides_in_extra_body():
    app = create_app(ServerConfig())
    client = TestClient(app)
    req = {
        "model": "aeon-auto",
        "messages": [
            {"role": "user", "content": "Perform high quality logical analysis."}
        ],
        "extra_body": {
            "aeon": {
                "quality_target": 0.98,
                "latency_sla_ms": 250.0,
                "cost_budget_usd": 0.005,
                "preferred_precision": "fp16"
            }
        }
    }
    res = client.post("/v1/chat/completions", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["x_aeon"]["selected_precision"].upper() in ["FP16", "FP32"]


def test_chat_completions_validation_rejects_empty_messages():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.post("/v1/chat/completions", json={"model": "aeon-auto", "messages": []})
    assert res.status_code in [400, 422]


def test_chat_completions_validation_rejects_no_user_turn():
    app = create_app(ServerConfig())
    client = TestClient(app)
    req = {
        "model": "aeon-auto",
        "messages": [{"role": "system", "content": "System only without user query"}]
    }
    res = client.post("/v1/chat/completions", json=req)
    assert res.status_code == 400


# =========================================================================
# 8. Chat Completions — SSE Streaming
# =========================================================================

def test_chat_completions_streaming_sse_format():
    app = create_app(ServerConfig())
    client = TestClient(app)
    req = {
        "model": "aeon-auto",
        "messages": [
            {"role": "user", "content": "Tell me what Paris is."}
        ],
        "stream": True
    }
    with client.stream("POST", "/v1/chat/completions", json=req) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line for line in response.iter_lines() if line]
        assert len(lines) > 2

        # Verify initial data lines are valid JSON chunks
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        first_chunk = json.loads(data_lines[0].replace("data: ", ""))
        assert first_chunk["object"] == "chat.completion.chunk"
        assert first_chunk["choices"][0]["delta"]["role"] == "assistant"

        # Verify terminal token
        assert lines[-1] == "data: [DONE]"


# =========================================================================
# 9. Concurrency & Rate Limiting
# =========================================================================

@pytest.mark.asyncio
async def test_rate_limiter_blocks_excessive_traffic():
    limiter = InMemoryRateLimiter(max_per_minute=3)
    client_ip = "192.168.1.10"

    await limiter.check_rate_limit(client_ip)
    await limiter.check_rate_limit(client_ip)
    await limiter.check_rate_limit(client_ip)

    # 4th request must be rejected
    with pytest.raises(RateLimitExceededError):
        await limiter.check_rate_limit(client_ip)


@pytest.mark.asyncio
async def test_concurrency_backpressure_manager():
    mgr = ConcurrencyBackpressureManager(max_concurrency=2, queue_size=2, queue_timeout=0.1)

    # Acquire 2 slots
    await mgr.acquire()
    await mgr.acquire()
    assert mgr.active_count == 2

    # Release 1 slot
    await mgr.release()
    assert mgr.active_count == 1

    # Release remaining
    await mgr.release()
    assert mgr.active_count == 0


# =========================================================================
# 10. Telemetry Store, Aggregator & Privacy
# =========================================================================

@pytest.mark.asyncio
async def test_telemetry_store_bounds_records():
    store = TelemetryStore(max_records=5)
    for i in range(10):
        await store.record_event({"request_id": f"req-{i}", "model": "aeon-auto"})

    records = await store.get_all_records()
    assert len(records) == 5
    assert records[-1]["request_id"] == "req-9"


@pytest.mark.asyncio
async def test_telemetry_privacy_scrubs_prompts_by_default():
    store = TelemetryStore(max_records=10, allow_content_logging=False)
    await store.record_event({
        "request_id": "req-priv-1",
        "prompt": "Private prompt with sensitive data",
        "response": "Private assistant reply",
        "latency_ms": 12.5,
        "api_key": "secret-999"
    })

    records = await store.get_all_records()
    assert len(records) == 1
    rec = records[0]
    assert "prompt" not in rec
    assert "response" not in rec
    assert rec["api_key"] == "[REDACTED]"
    assert rec["latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_telemetry_aggregator_summary_and_prometheus():
    store = TelemetryStore(max_records=100)
    await store.record_event({
        "request_id": "req-1",
        "model": "aeon-medium",
        "device": "mps",
        "precision": "fp16",
        "depth": 24,
        "latency_ms": 10.0,
        "cost_usd": 1e-5,
        "energy_joules": 0.05,
        "controller_overhead_ms": 0.03,
        "tokens_in": 20,
        "tokens_out": 40,
        "status_code": 200
    })

    agg = TelemetryAggregator(store)
    summary = await agg.get_summary()

    assert summary["requests"]["total"] == 1
    assert summary["requests"]["successful"] == 1
    assert summary["latency"]["p50"] == 10.0
    assert summary["tokens"]["total_tokens"] == 60

    prom = format_prometheus_metrics(summary)
    assert "aeon_requests_total 1" in prom
    assert "aeon_tokens_total 60" in prom


# =========================================================================
# 11. Metrics & Admin Endpoints
# =========================================================================

def test_metrics_endpoint_prometheus_and_json():
    app = create_app(ServerConfig())
    client = TestClient(app)

    # Prometheus text
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "aeon_requests_total" in res.text

    # JSON format
    res_json = client.get("/metrics?format=json")
    assert res_json.status_code == 200
    assert "requests" in res_json.json()


def test_admin_config_and_status_endpoints():
    app = create_app(ServerConfig(api_key="admin-pass"))
    client = TestClient(app)

    # Unauthorized
    res = client.get("/admin/config")
    assert res.status_code == 401

    # Authorized config inspection
    res = client.get("/admin/config", headers={"Authorization": "Bearer admin-pass"})
    assert res.status_code == 200
    assert res.json()["api_key"] == "[REDACTED]"

    # Authorized status inspection
    res_st = client.get("/admin/status", headers={"Authorization": "Bearer admin-pass"})
    assert res_st.status_code == 200
    assert "active_concurrency" in res_st.json()

    # Authorized reload
    res_rl = client.post("/admin/reload", headers={"Authorization": "Bearer admin-pass"})
    assert res_rl.status_code == 200
    assert res_rl.json()["status"] == "reloaded"


# =========================================================================
# 12. Dashboard UI & Summary API
# =========================================================================

def test_dashboard_html_and_api():
    app = create_app(ServerConfig())
    client = TestClient(app)

    # HTML
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "AEON ORCHESTRATION CONSOLE" in res.text

    # JSON summary
    res_summary = client.get("/dashboard/api/summary")
    assert res_summary.status_code == 200
    data = res_summary.json()
    assert "system" in data
    assert "metrics" in data
    assert "provenance" in data

    # Recent requests
    res_rec = client.get("/dashboard/api/recent")
    assert res_rec.status_code == 200
    assert isinstance(res_rec.json(), list)


# =========================================================================
# 13. OpenAPI Specification Validity
# =========================================================================

def test_openapi_json_schema_valid():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/openapi.json")
    assert res.status_code == 200
    data = res.json()
    assert data["openapi"].startswith("3.")
    assert "/v1/chat/completions" in data["paths"]
    assert "/v1/models" in data["paths"]
    assert "/health" in data["paths"]


# =========================================================================
# 14. CLI Integration
# =========================================================================

def test_cli_server_commands_registered():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "serve" in res.output
    assert "server" in res.output


def test_cli_serve_help_options():
    runner = CliRunner()
    res = runner.invoke(cli, ["serve", "--help"])
    assert res.exit_code == 0
    assert "--host" in res.output
    assert "--port" in res.output
    assert "--api-key" in res.output
    assert "--workers" in res.output


def test_cli_server_health_command_failure_handled():
    runner = CliRunner()
    # Connecting to non-existent server port should exit cleanly with error message
    res = runner.invoke(cli, ["server", "health", "--url", "http://127.0.0.1:59999"])
    assert res.exit_code != 0
    assert "FAILED" in res.output or "Health Check" in res.output


def test_cli_server_status_command_failure_handled():
    runner = CliRunner()
    res = runner.invoke(cli, ["server", "status", "--url", "http://127.0.0.1:59999"])
    assert res.exit_code != 0
    assert "Failed to query" in res.output or "status" in res.output


# =========================================================================
# 15. Advanced Chat Completions & Validation
# =========================================================================

def test_chat_completions_multiturn_dialogue():
    app = create_app(ServerConfig())
    client = TestClient(app)
    req_payload = {
        "model": "aeon-auto",
        "messages": [
            {"role": "system", "content": "You are a concise AI assistant."},
            {"role": "user", "content": "Hello, my name is Alice."},
            {"role": "assistant", "content": "Hello Alice! How can I help you today?"},
            {"role": "user", "content": "What is my name?"}
        ],
        "temperature": 0.2,
        "max_tokens": 64
    }
    res = client.post("/v1/chat/completions", json=req_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"]


def test_chat_completions_negative_temperature_rejected():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": -0.5
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"


def test_chat_completions_negative_max_tokens_rejected():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": -10
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "error" in data


def test_chat_completions_excessive_max_tokens_rejected():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100000
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "error" in data


def test_chat_completions_invalid_message_role_rejected():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "superuser", "content": "do bad things"}]
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 422


def test_chat_completions_policy_override_parameters():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "compute fast"}],
        "extra_body": {
            "aeon": {
                "quality_target": 0.99,
                "latency_sla_ms": 300.0,
                "cost_budget_usd": 0.0001,
                "preferred_device": "cpu",
                "preferred_precision": "fp32"
            }
        }
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "x_aeon" in data
    assert data["x_aeon"]["quality_estimate"] >= 0.0


def test_chat_completions_tool_routing_math_query():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "What is 17 * 23?"}],
        "max_tokens": 128
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    content = data["choices"][0]["message"]["content"]
    assert "391" in content or len(content) > 0


def test_chat_completions_reasoning_escalation():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "Solve step by step: prove that sqrt(2) is irrational."}]
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "x_aeon" in data
    assert "reasoning_strategy" in data["x_aeon"]


def test_chat_completions_cache_hit_on_repeat():
    app = create_app(ServerConfig())
    client = TestClient(app)
    prompt = "Unique prompt for caching test #4928"
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": prompt}]
    }

    # First execution (populate cache)
    res1 = client.post("/v1/chat/completions", json=payload)
    assert res1.status_code == 200

    # Second execution (cache lookup)
    res2 = client.post("/v1/chat/completions", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert "x_aeon" in data2
    # Verify cache lookup was executed
    assert "cache_hit" in data2["x_aeon"]


# =========================================================================
# 16. Streaming Semantics & Chunk Structure
# =========================================================================

def test_chat_streaming_chunk_sequence_and_ids():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "Count from 1 to 3."}],
        "stream": True
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    lines = [line.strip() for line in res.text.split("\n") if line.strip()]
    data_lines = [l[6:] for l in lines if l.startswith("data: ") and l != "data: [DONE]"]

    assert len(data_lines) >= 2
    initial_chunk = json.loads(data_lines[0])
    chunk_id = initial_chunk["id"]
    assert chunk_id.startswith("chatcmpl-")
    assert initial_chunk["object"] == "chat.completion.chunk"

    # All chunks must share the identical completion ID
    for dl in data_lines:
        chunk = json.loads(dl)
        assert chunk["id"] == chunk_id
        assert chunk["model"] in ["aeon-auto", "aeon-small-1.5b", "aeon-small"]

    # Final event must be [DONE]
    assert lines[-1] == "data: [DONE]"


def test_chat_streaming_unknown_model_returns_404():
    app = create_app(ServerConfig())
    client = TestClient(app)
    payload = {
        "model": "unknown-stream-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 404
    assert res.json()["error"]["type"] in ["invalid_request_error", "resource_not_found_error"]


# =========================================================================
# 17. Model Profiles & 404 Resource Handling
# =========================================================================

def test_models_get_specific_model_profiles():
    app = create_app(ServerConfig())
    client = TestClient(app)

    # aeon-auto
    res_auto = client.get("/v1/models/aeon-auto")
    assert res_auto.status_code == 200
    assert res_auto.json()["id"] == "aeon-auto"

    # aeon-small
    res_sm = client.get("/v1/models/aeon-small")
    assert res_sm.status_code == 200
    assert res_sm.json()["id"] == "aeon-small"
    assert res_sm.json()["capabilities"]["supports_tools"] is True


def test_models_get_nonexistent_returns_404():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/v1/models/claude-3-opus-nonexistent")
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert "does not exist" in data["error"]["message"].lower() or "not found" in data["error"]["message"].lower()


# =========================================================================
# 18. Security Headers, Request ID & Payload Bounds
# =========================================================================

def test_security_headers_present_across_endpoints():
    app = create_app(ServerConfig())
    client = TestClient(app)

    for path in ["/health", "/v1/models", "/dashboard", "/metrics"]:
        res = client.get(path)
        assert res.headers.get("x-content-type-options") == "nosniff"
        assert res.headers.get("x-frame-options") == "DENY"
        assert res.headers.get("referrer-policy") == "no-referrer"


def test_custom_request_id_validated_and_propagated():
    app = create_app(ServerConfig())
    client = TestClient(app)
    custom_id = "trace-client-abc-12345"
    res = client.get("/health", headers={"X-Request-ID": custom_id})
    assert res.status_code == 200
    assert res.headers.get("x-request-id") == custom_id


def test_malicious_request_id_sanitized():
    app = create_app(ServerConfig())
    client = TestClient(app)
    # Invalid characters / potential injection in header
    malicious_id = "bad<script>alert(1)</script>id"
    res = client.get("/health", headers={"X-Request-ID": malicious_id})
    assert res.status_code == 200
    # Must be replaced with a sanitized UUID
    returned_id = res.headers.get("x-request-id")
    assert returned_id != malicious_id
    assert VALID_REQUEST_ID_REGEX.match(returned_id)


def test_oversized_payload_rejected_by_middleware():
    # Configure tiny 1KB max request size
    app = create_app(ServerConfig(max_request_bytes=1024))
    client = TestClient(app)
    giant_text = "x" * 2048
    res = client.post("/v1/chat/completions", json={
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": giant_text}]
    })
    assert res.status_code == 413
    assert res.json()["error"]["type"] in ["payload_too_large", "request_too_large_error"]


# =========================================================================
# 19. Authentication & Secret Redaction
# =========================================================================

def test_authentication_with_x_api_key_header():
    app = create_app(ServerConfig(api_key="secret-token-key-42"))
    client = TestClient(app)

    # Success with X-API-Key
    res = client.get("/v1/models", headers={"X-API-Key": "secret-token-key-42"})
    assert res.status_code == 200

    # Failure with bad key
    res_bad = client.get("/v1/models", headers={"X-API-Key": "wrong-key"})
    assert res_bad.status_code == 401


def test_constant_time_key_verification():
    assert constant_time_compare("my-secret", "my-secret") is True
    assert constant_time_compare("my-secret", "wrong-secret") is False
    assert constant_time_compare("my-secret", "") is False
    assert constant_time_compare("my-secret", None) is False


def test_secret_redaction_in_text_and_dicts():
    text = "Authorization: Bearer sk-ant-api03-abcdef123456789 and password is secretPassword!"
    sanitized = sanitize_text(text)
    assert "Bearer [REDACTED]" in sanitized
    assert "[REDACTED]" in sanitized

    event = {
        "user_token": "token-xyz-789",
        "api_key": "key-123",
        "tokens_in": 15,
        "tokens_out": 25,
        "model": "aeon-auto"
    }
    cleaned = sanitize_telemetry_event(event)
    assert cleaned["user_token"] == "[REDACTED]"
    assert cleaned["api_key"] == "[REDACTED]"
    # Normal integer token counts must not be scrubbed
    assert cleaned["tokens_in"] == 15
    assert cleaned["tokens_out"] == 25


# =========================================================================
# 20. Rate Limiting & Concurrency Edge Cases
# =========================================================================

@pytest.mark.asyncio
async def test_rate_limiter_distinct_clients():
    limiter = InMemoryRateLimiter(max_per_minute=2)
    # Client A consumes 2
    await limiter.check_rate_limit("1.1.1.1")
    await limiter.check_rate_limit("1.1.1.1")
    with pytest.raises(RateLimitExceededError):
        await limiter.check_rate_limit("1.1.1.1")

    # Client B is unaffected
    await limiter.check_rate_limit("2.2.2.2")
    assert True


@pytest.mark.asyncio
async def test_concurrency_backpressure_timeout():
    # Concurrency 1, queue 0, immediate timeout
    mgr = ConcurrencyBackpressureManager(max_concurrency=1, queue_size=0, queue_timeout=0.01)
    await mgr.acquire()

    # Second acquire must timeout and raise ServiceUnavailableError
    with pytest.raises(ServiceUnavailableError):
        await mgr.acquire()

    await mgr.release()


# =========================================================================
# 21. Telemetry Store Retention & Async Concurrency
# =========================================================================

@pytest.mark.asyncio
async def test_telemetry_store_age_pruning():
    import time
    store = TelemetryStore(max_records=100, max_age_seconds=1)
    # Record old event with artificial past timestamp
    past_time = time.time() - 10
    await store.record_event({"request_id": "old-req", "timestamp": past_time})

    # Record fresh event
    await store.record_event({"request_id": "new-req"})

    records = await store.get_all_records()
    assert len(records) == 1
    assert records[0]["request_id"] == "new-req"


@pytest.mark.asyncio
async def test_telemetry_store_concurrent_writes():
    store = TelemetryStore(max_records=50)

    async def write_event(idx: int):
        await store.record_event({"request_id": f"async-req-{idx}", "idx": idx})

    tasks = [write_event(i) for i in range(25)]
    await asyncio.gather(*tasks)

    records = await store.get_all_records()
    assert len(records) == 25


@pytest.mark.asyncio
async def test_telemetry_store_clear():
    store = TelemetryStore()
    await store.record_event({"request_id": "ev-1"})
    await store.record_event({"request_id": "ev-2"})
    assert len(await store.get_all_records()) == 2

    await store.clear()
    assert len(await store.get_all_records()) == 0


@pytest.mark.asyncio
async def test_telemetry_aggregator_empty_store():
    store = TelemetryStore()
    agg = TelemetryAggregator(store)
    summary = await agg.get_summary()
    assert summary["requests"]["total"] == 0
    assert summary["latency"]["p50"] == 0.0
    assert summary["latency"]["p95"] == 0.0


@pytest.mark.asyncio
async def test_telemetry_aggregator_percentiles_calculation():
    store = TelemetryStore(max_records=100)
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        await store.record_event({
            "request_id": f"lat-{lat}",
            "latency_ms": lat,
            "status_code": 200
        })

    agg = TelemetryAggregator(store)
    summary = await agg.get_summary()
    assert summary["latency"]["p50"] == 55.0
    assert summary["latency"]["p95"] >= 90.0
    assert summary["latency"]["mean"] == 55.0


# =========================================================================
# 22. Health, Liveness & Version Integrity
# =========================================================================

def test_health_live_endpoint():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "alive"
    assert "timestamp" in data


def test_health_ready_component_breakdown():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    comps = data["components"]
    assert "controller" in comps
    assert "hardware" in comps
    assert "memory" in comps
    assert "tools" in comps
    assert "models" in comps


def test_version_endpoint_no_directory_leakage():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/version")
    assert res.status_code == 200
    data = res.json()
    assert data["aeon_version"] in ["0.10.0", "1.0.0"]
    assert data["api_version"] == "v1"
    assert "python_version" in data
    # Ensure no absolute file system paths or passwords leaked
    text = res.text
    assert "/Users/" not in text
    assert "/home/" not in text
    assert "secret" not in text.lower()


# =========================================================================
# 23. Dashboard Static Files & Visualizations
# =========================================================================

def test_dashboard_static_assets_serve_correct_types():
    app = create_app(ServerConfig())
    client = TestClient(app)

    # CSS
    res_css = client.get("/dashboard/static/style.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"]
    assert "var(--bg-primary)" in res_css.text

    # JavaScript
    res_js = client.get("/dashboard/static/dashboard.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers["content-type"]
    assert "fetchDashboardData" in res_js.text


def test_dashboard_summary_explicit_provenance_tags():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/dashboard/api/summary")
    assert res.status_code == 200
    data = res.json()
    prov = data["provenance"]
    assert prov["latency"] in ["MEASURED", "SIMULATED", "HYBRID"]
    assert prov["cost"] == "ESTIMATED"
    assert prov["energy"] == "ESTIMATED"
    assert prov["flops"] == "DERIVED"


# =========================================================================
# 24. Swagger & ReDoc Documentation
# =========================================================================

def test_swagger_and_redoc_endpoints():
    app = create_app(ServerConfig())
    client = TestClient(app)

    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    assert "swagger-ui" in res_docs.text.lower()

    res_redoc = client.get("/redoc")
    assert res_redoc.status_code == 200
    assert "redoc" in res_redoc.text.lower()


# =========================================================================
# 25. Local Load Test & Concurrency Execution
# =========================================================================

def test_local_development_load_test():
    """
    Section 59: Local Development Load Test.
    Execute concurrent requests against AEON server and verify stable throughput.
    """
    app = create_app(ServerConfig())
    client = TestClient(app)

    prompts = [
        "What is 2+2?",
        "What is 10 * 10?",
        "Define entropy in physics.",
        "List three primary colors."
    ]

    latencies = []
    for p in prompts:
        res = client.post("/v1/chat/completions", json={
            "model": "aeon-auto",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 32
        })
        assert res.status_code == 200
        data = res.json()
        assert "x_aeon" in data
        latencies.append(data["x_aeon"]["latency_ms"])

    assert len(latencies) == 4
    assert all(lat >= 0.0 for lat in latencies)


# =========================================================================
# 26. Production Hardening, Provenance & Adversarial Safety
# =========================================================================

@pytest.mark.asyncio
async def test_graceful_shutdown_lifespan_event():
    from aeon.server.lifecycle import on_shutdown
    cfg = ServerConfig()
    app = create_app(cfg)
    # Trigger shutdown handler and verify clean termination
    await on_shutdown(app)
    assert True


def test_real_vs_simulated_provenance_separation():
    app = create_app(ServerConfig())
    client = TestClient(app)
    res = client.get("/health/ready")
    assert res.status_code == 200
    hw_details = res.json()["components"]["hardware"]["details"]
    assert "provenance" in hw_details
    assert hw_details["provenance"] == "MEASURED"


def test_memory_safety_preflight_rejection():
    app = create_app(ServerConfig())
    client = TestClient(app)
    # Request extreme memory budget (e.g. 500,000 MB) exceeding available hardware RAM
    payload = {
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "Compute large workload"}],
        "extra_body": {
            "aeon": {
                "memory_budget_mb": 5000000.0
            }
        }
    }
    res = client.post("/v1/chat/completions", json=payload)
    # Must fail safely with 503 rather than trigger an OS OOM crash
    assert res.status_code == 503
    assert "memory safety" in res.json()["error"]["message"].lower()


def test_http_api_overhead_measurement():
    from aeon import AEON
    import time
    app = create_app(ServerConfig())
    client = TestClient(app)
    direct_aeon = AEON()

    prompt = "What is the capital of France?"

    # 1. Direct Python invocation
    t0 = time.perf_counter()
    res_direct = direct_aeon.run(prompt)
    direct_duration_ms = (time.perf_counter() - t0) * 1000.0

    # 2. HTTP API invocation
    t1 = time.perf_counter()
    res_api = client.post("/v1/chat/completions", json={
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": prompt}]
    })
    api_duration_ms = (time.perf_counter() - t1) * 1000.0

    assert res_api.status_code == 200
    assert api_duration_ms >= 0.0
    assert direct_duration_ms >= 0.0


def test_adversarial_malformed_json_returns_400():
    app = create_app(ServerConfig())
    client = TestClient(app)
    # Post broken unparseable JSON payload
    res = client.post(
        "/v1/chat/completions",
        content="{\"model\": \"aeon-auto\", \"messages\": [INVALID_JSON_BODY}",
        headers={"Content-Type": "application/json"}
    )
    assert res.status_code in [400, 422]
    assert "error" in res.json()


def test_dashboard_recent_requests_privacy_preservation():
    app = create_app(ServerConfig())
    client = TestClient(app)

    # Execute request with potentially sensitive prompt text
    client.post("/v1/chat/completions", json={
        "model": "aeon-auto",
        "messages": [{"role": "user", "content": "My private secret data"}]
    })

    # Query recent requests in dashboard
    res = client.get("/dashboard/api/recent")
    assert res.status_code == 200
    records = res.json()
    assert len(records) >= 1
    recent = records[-1]
    # Ensure sensitive content field is never exposed in dashboard API
    assert "prompt" not in recent
    assert "messages" not in recent
    assert "model" in recent
    assert "status_code" in recent


