# AEON — Adaptive Economically-Optimized Network

> **AEON is an adaptive compute orchestration layer that dynamically routes inference across available models, tools, memory, reasoning strategies, execution depth, precision and hardware.**

AEON is designed to solve each task at the lowest necessary computational cost while respecting quality, latency, memory, energy, and hardware constraints.

AEON is **NOT**:
- a new foundation model
- an OpenAI clone or OpenAI API dependency
- a hosted cloud service requirement
- a benchmark-only framework
- a claim of AGI

---

## Key Capabilities

1. **Dynamic Model Routing**: Routes queries across small, medium, and large models based on task complexity and Quality-per-Dollar optimization.
2. **Deterministic Tool Execution**: Diverts arithmetic, mathematical proofs, and structured operations to lightweight sandboxed tools (e.g., Calculator, Python sandbox, SymPy).
3. **Multi-Tier Memory Hierarchy**: Reuses exact and semantic memory across queries to bypass redundant model executions.
4. **Adaptive Latent & Explicit Reasoning**: Scales reasoning depth dynamically based on task difficulty.
5. **Adaptive Depth & Mixture-of-Depths**: Exits early when high confidence is established, saving unnecessary layer FLOPs.
6. **Hardware-Aware Precision & Quantization**: Adapts execution precision (FP32, FP16, INT8, FP8) and device placement (CPU, Apple Silicon MPS, CUDA).
7. **OpenAPI Server & Web Dashboard**: Exposes an OpenAI-compatible HTTP interface (`/v1/chat/completions`, `/v1/models`) and an observable, zero-dependency web dashboard (`/dashboard`).

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/aeon.git
cd aeon
pip install -e .
```

### Starting the Server

```bash
# Starts on http://127.0.0.1:8000
aeon serve

# Optional: Require an API key
aeon serve --api-key my-secure-token
```

### Interacting via `curl`

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

### Interacting via Official OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="aeon-local"  # Dummy key accepted when auth is disabled
)

response = client.chat.completions.create(
    model="aeon-auto",
    messages=[{"role": "user", "content": "Explain Amdahl's Law concisely."}]
)

print(response.choices[0].message.content)
```

### Web Dashboard

Visit `http://127.0.0.1:8000/dashboard` in your browser to view real-time system metrics, model distribution, latency percentiles, tool usage, and economic telemetry with full empirical provenance.

---

## Documentation

- [Server Guide](docs/AEON_SERVER.md)
- [API Reference](docs/API.md)
- [Dashboard Manual](docs/DASHBOARD.md)
- [Phase 10 Final Report](docs/AEON_PHASE10_REPORT.md)
- [OpenAPI Specification](docs/openapi.json)

---

## License

Apache 2.0
