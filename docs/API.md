# AEON API Reference

## Overview

The AEON HTTP API provides an OpenAPI-compliant, drop-in replacement for OpenAI Chat Completions. It allows existing OpenAI client libraries, LangChain, LlamaIndex, and curl scripts to run against AEON without needing paid API keys or OpenAI infrastructure.

Base URL: `http://127.0.0.1:8000/v1`

---

## 1. Endpoints Summary

| Method | Path | Summary | Auth Required |
| :--- | :--- | :--- | :--- |
| `POST` | `/v1/chat/completions` | Execute adaptive chat completion (stream / non-stream) | If `AEON_API_KEY` set |
| `GET` | `/v1/models` | List all available models and orchestration profiles | If `AEON_API_KEY` set |
| `GET` | `/v1/models/{model_id}` | Retrieve capability metadata for a specific model | If `AEON_API_KEY` set |
| `GET` | `/health` | Basic service health | No |
| `GET` | `/health/live` | Kubernetes liveness probe | No |
| `GET` | `/health/ready` | Kubernetes readiness probe (subsystem status) | No |
| `GET` | `/version` | Version, build, and environment information | No |
| `GET` | `/metrics` | Prometheus or JSON metrics exposition | No |
| `GET` | `/admin/config` | View active server configuration (secrets redacted) | Yes (if enabled) |
| `GET` | `/admin/status` | Query engine status and memory headroom | Yes (if enabled) |
| `POST` | `/admin/reload` | Reload runtime state and reset memory cache | Yes (if enabled) |
| `GET` | `/dashboard` | Interactive Web Dashboard Console | No |
| `GET` | `/openapi.json` | Complete OpenAPI 3.1 specification schema | No |
| `GET` | `/docs` | Swagger UI interactive documentation | No |
| `GET` | `/redoc` | ReDoc API documentation | No |

---

## 2. Chat Completions (`POST /v1/chat/completions`)

### Request Schema

```json
{
  "model": "aeon-auto",
  "messages": [
    {
      "role": "system",
      "content": "You are a concise AI assistant."
    },
    {
      "role": "user",
      "content": "What is 17 * 23?"
    }
  ],
  "temperature": 0.0,
  "max_tokens": 256,
  "stream": false,
  "extra_body": {
    "aeon": {
      "quality_target": 0.95,
      "latency_sla_ms": 500.0,
      "cost_budget_usd": 0.001,
      "preferred_device": "auto",
      "preferred_precision": "auto"
    }
  }
}
```

### Response Schema (Non-Streaming)

```json
{
  "id": "chatcmpl-aeon-1700000000000-abcd",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "aeon-small-1.5b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "391"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 4,
    "total_tokens": 16
  },
  "x_aeon": {
    "request_id": "req-1700000000000",
    "selected_model": "aeon-small-1.5b",
    "selected_device": "mps",
    "selected_precision": "fp16",
    "selected_depth": 16,
    "selected_strategy": "DIRECT",
    "cache_hit": false,
    "tool_used": "calculator",
    "reasoning_strategy": "direct",
    "speculation_used": true,
    "quality_estimate": 0.98,
    "latency_ms": 14.2,
    "estimated_cost_usd": 0.0000021,
    "estimated_energy_joules": 0.041,
    "controller_overhead_ms": 0.038,
    "provenance": "HYBRID"
  }
}
```

### Streaming SSE (`stream: true`)

Streaming chunks are emitted using standard Server-Sent Events (`text/event-stream`):

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"aeon-small-1.5b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"aeon-small-1.5b","choices":[{"index":0,"delta":{"content":"391"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1700000000,"model":"aeon-small-1.5b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## 3. Client Examples

### Using `curl`

#### Example 1: Deterministic Math Query (Tool Path)

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aeon-auto",
    "messages": [
      {"role": "user", "content": "What is 17 * 23?"}
    ]
  }'
```

#### Example 2: Reasoning Escalation Query

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aeon-auto",
    "messages": [
      {"role": "user", "content": "Solve step by step: prove that sqrt(2) is irrational."}
    ]
  }'
```

#### Example 3: Constrained Request with Policy Overrides

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aeon-auto",
    "messages": [
      {"role": "user", "content": "Summarize thermodynamics laws."}
    ],
    "extra_body": {
      "aeon": {
        "quality_target": 0.95,
        "latency_sla_ms": 300.0,
        "cost_budget_usd": 0.0001
      }
    }
  }'
```

#### Example 4: Server-Sent Events Streaming

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aeon-auto",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "stream": true
  }'
```

---

### Using Python `openai` SDK

```python
from openai import OpenAI

# Point client to local AEON server
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="aeon-local"  # Required by SDK syntax; ignored by AEON if AEON_API_KEY is not set
)

# 1. Non-streaming completion
response = client.chat.completions.create(
    model="aeon-auto",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Amdahl's Law concisely."}
    ],
    temperature=0.0,
    max_tokens=150,
    extra_body={
        "aeon": {
            "quality_target": 0.95,
            "latency_sla_ms": 500.0
        }
    }
)
print("Response:", response.choices[0].message.content)

# 2. Streaming completion
stream = client.chat.completions.create(
    model="aeon-auto",
    messages=[
        {"role": "user", "content": "Explain speculative decoding in 3 bullet points."}
    ],
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
```

---

## 4. Models API (`GET /v1/models`)

Returns all currently available AEON model profiles and routing options:

```bash
curl http://127.0.0.1:8000/v1/models
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "aeon-auto",
      "object": "model",
      "created": 1700000000,
      "owned_by": "aeon",
      "capabilities": {
        "adaptive_routing": true,
        "dynamic_depth": true,
        "multi_tier_memory": true,
        "tool_routing": true,
        "speculative_decoding": true,
        "hardware_aware": true,
        "supports_streaming": true
      }
    },
    {
      "id": "aeon-small-1.5b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "aeon",
      "capabilities": {
        "parameter_count_b": 1.5,
        "context_window": 8192,
        "quality_score": 0.86,
        "supports_streaming": true,
        "supported_precisions": ["fp32", "fp16", "int8", "int4"],
        "supports_tools": true,
        "supports_reasoning": true
      }
    }
  ]
}
```

---

## 5. Metrics API (`GET /metrics`)

### Prometheus Exposition (Default)

```bash
curl http://127.0.0.1:8000/metrics
```

Output:

```text
# HELP aeon_requests_total Total number of inference requests
# TYPE aeon_requests_total counter
aeon_requests_total 42
aeon_requests_success 42
aeon_requests_failed 0
aeon_tokens_total 1840
aeon_cache_hit_rate 0.14
aeon_latency_p50_ms 14.2
aeon_latency_p95_ms 38.6
aeon_estimated_cost_usd_total 0.000084
```

### JSON Format

```bash
curl http://127.0.0.1:8000/metrics?format=json
```

---

## 6. Error Handling

All errors adhere to OpenAI error response formatting:

```json
{
  "error": {
    "message": "Model 'gpt-5' does not exist or is not available.",
    "type": "resource_not_found_error",
    "param": "model",
    "code": "resource_not_found"
  }
}
```

### Supported HTTP Status Codes

- `400 Bad Request`: Malformed JSON or unparseable input.
- `401 Unauthorized`: Missing or invalid API key when authentication is configured.
- `403 Forbidden`: Authenticated user lacks permission.
- `404 Not Found`: Model or resource identifier not registered.
- `413 Payload Too Large`: Request body exceeds `max_request_bytes` (1MB default).
- `422 Unprocessable Content`: Pydantic validation failure (e.g., negative temperature, excessive max_tokens).
- `429 Too Many Requests`: Rate limiter exceeded per-minute or burst threshold.
- `500 Internal Server Error`: Generic internal failure (Python stack traces are scrubbed).
- `503 Service Unavailable`: Concurrency backpressure queue saturated or memory safety rejection.
