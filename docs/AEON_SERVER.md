# AEON Server Guide

## Overview

The AEON Server is an OpenAPI-specification-compatible HTTP inference service that wraps AEON's adaptive compute orchestration engine. It transforms AEON from a research/benchmark library into a local-first, provider-agnostic, drop-in replacement for OpenAI-compatible chat endpoints (`/v1/chat/completions` and `/v1/models`).

AEON acts as an **adaptive compute orchestration layer** that dynamically routes tasks across models, deterministic tools, multi-level memory, adaptive reasoning strategies, execution depth (early exit), numerical precision, and target hardware accelerators.

---

## 1. Quick Start

### Starting the Server via CLI

```bash
# Default: binds to 127.0.0.1:8000, auth disabled for local dev
aeon serve

# Custom host, port, and workers
aeon serve --host 127.0.0.1 --port 8080 --workers 2

# Require API key authentication
aeon serve --api-key my-secure-token-12345

# Using external configuration file
aeon serve --config config/server.json
```

### Checking Health and Status via CLI

```bash
# Check server readiness and component health
aeon server health --url http://127.0.0.1:8000

# Inspect server status and memory headroom
aeon server status --url http://127.0.0.1:8000 --api-key my-secure-token-12345
```

---

## 2. Configuration Hierarchy

Server configuration follows strict precedence:

```text
CLI Arguments > Environment Variables > Configuration File > Safe Defaults
```

### Configuration Parameters

| Parameter | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `host` | `AEON_HOST` | `127.0.0.1` | Network interface binding. Warning emitted if `0.0.0.0`. |
| `port` | `AEON_PORT` | `8000` | Port number (1–65535). |
| `workers` | `AEON_WORKERS` | `1` | Uvicorn worker process count. |
| `api_key` | `AEON_API_KEY` | `None` | Shared secret token. If null, auth is disabled for local dev. |
| `cors_allowed_origins` | `AEON_CORS_ORIGINS` | `[]` | Allowed CORS origins. Never defaults to `*`. |
| `max_concurrency` | `AEON_MAX_CONCURRENCY` | `16` | Maximum concurrent in-flight executions. |
| `queue_size` | `AEON_QUEUE_SIZE` | `64` | Backpressure queue depth. |
| `queue_timeout` | `AEON_QUEUE_TIMEOUT` | `10.0` | Seconds to wait before returning 503 overload. |
| `rate_limit_per_minute`| `AEON_RATE_LIMIT_PER_MINUTE` | `120` | Max requests per minute per IP address. |
| `max_request_bytes` | `AEON_MAX_REQUEST_BYTES` | `1048576` (1MB)| Maximum accepted HTTP request body size. |
| `max_messages` | `AEON_MAX_MESSAGES` | `100` | Maximum messages allowed in conversational history. |
| `max_max_tokens` | `AEON_MAX_MAX_TOKENS` | `8192` | Hard ceiling for `max_tokens`. |
| `telemetry_enabled` | `AEON_TELEMETRY_ENABLED` | `true` | Bounded in-memory telemetry recording. |
| `telemetry_max_records`| `AEON_TELEMETRY_MAX_RECORDS`| `1000` | Ring buffer size for telemetry events. |
| `allow_content_logging`| `AEON_ALLOW_CONTENT_LOGGING`| `false` | When false, all prompt and response text is scrubbed. |
| `dashboard_enabled` | `AEON_DASHBOARD_ENABLED` | `true` | Mounts `/dashboard` console. |

---

## 3. Security Architecture

1. **Local-First Binding**:
   The server defaults strictly to `127.0.0.1`. If configured to bind to a non-loopback address (e.g. `0.0.0.0`), a critical security warning is printed to `stderr` urging the administrator to set an `api_key`.
2. **Defensive Headers**:
   Every response includes:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `Referrer-Policy: no-referrer`
3. **Constant-Time Authentication**:
   API keys are compared via `hmac.compare_digest` to prevent timing side-channel attacks. Keys can be transmitted via `Authorization: Bearer <token>` or `X-API-Key: <token>`.
4. **Log & Telemetry Redaction**:
   Sensitive patterns (API keys, authorization headers, passwords, session tokens) are filtered via regex scrubbing before logging or telemetry storage.
5. **No Arbitrary Code Execution**:
   API requests cannot pass arbitrary Python or shell code to tools. The tool architecture is locked to predefined sandboxed tools (Calculator, Python sandbox, SymPy, Retrieval).

---

## 4. Backend Architecture & Provenance

AEON supports both real local accelerators (Apple Silicon MPS, CUDA, CPU) and analytical simulation kernels (A100, H100):

- **Measured Execution**: Local CPU and Apple Silicon MPS runs use measured execution latency and real memory usage.
- **Simulated Execution**: Datacenter profiles (A100 SXM4 80GB, H100 SXM5 80GB) are analytical benchmark simulations.
- **Provenance Integrity**: All responses, dashboard cards, and telemetry records include explicit provenance labels (`MEASURED`, `SIMULATED`, `HYBRID`, `ESTIMATED`, `DERIVED`). Simulated numbers are never misrepresented as physical hardware measurements.

---

## 5. Graceful Shutdown & Backpressure

- On `SIGTERM` or `SIGINT`, the server stops accepting new connections, allows in-flight requests to conclude within the shutdown timeout, flushes telemetry, clears cache structures, and terminates without orphan tasks.
- When concurrent request volume exceeds `max_concurrency` and the queue fills beyond `queue_size`, excess requests receive an immediate `503 Service Unavailable` with descriptive backpressure headers, preventing unconstrained memory growth or Out-of-Memory crashes.
