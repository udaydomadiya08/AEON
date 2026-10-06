# AEON

> **AEON is an adaptive compute orchestration layer that dynamically chooses models, tools, memory, reasoning, inference depth, precision, hardware, and verification strategies to optimize the quality–compute–latency–cost tradeoff.**

---

## What is AEON?

AEON (**Adaptive Economically-Optimized Network**) is a local-first, provider-agnostic runtime and inference orchestration layer. It wraps AI workloads to adaptively allocate computational resources according to task difficulty, latency SLAs, memory constraints, and budget targets.

AEON is **NOT**:
* A new foundation model or pre-trained weight set.
* An OpenAI clone, proxy, or wrapper around paid OpenAI APIs.
* A hosted cloud service dependency.
* A claim of AGI or general artificial superintelligence.
* A benchmark-only synthetic framework.

AEON operates as an intelligent operating layer that dynamically decides:
1. **Which execution route** (deterministic tool, cached memory, small model, medium model, or large model) to select.
2. **Which inference depth** (full depth vs. adaptive early layer exit) to execute.
3. **Which reasoning strategy** (direct response, latent planning, or multi-step verification) to apply.
4. **Which numerical precision and hardware accelerator** (CPU, Apple Silicon MPS, or CUDA) to target.

---

## Why AEON?

Standard LLM inference suffers from static over-allocation: every prompt—whether a trivial single-digit calculation, an identical repeat query, or an intricate multi-step mathematical proof—runs through the same parameter count, full network layer depth, and fixed precision.

AEON introduces **Value-of-Information (VOI)** and economic constraint management to dynamically right-size compute expenditure:
* Simple arithmetic and fact lookups route to deterministic sandboxed tools or exact memory caches in sub-millisecond time.
* Standard questions route to lightweight draft or small models with early-exit layers.
* High-complexity reasoning tasks escalate dynamically to deep models and verification loops.

---

## Architecture

The AEON execution pipeline evaluates tasks through an end-to-end, multi-tier decision DAG before generation:

```text
Incoming Request
      ↓
[ Task Classification & Difficulty Estimation ]
      ↓
[ Multi-Tier Memory Lookup (L1 Exact / L4 Semantic) ] ── (Cache Hit) ──→ Response
      ↓ (Miss)
[ Deterministic Tool Routing (Calculator / SymPy / Python) ] ── (Tool Match) ──→ Response
      ↓ (Model Required)
[ Adaptive Reasoning Selection (Direct / Latent / Explicit) ]
      ↓
[ Speculative Draft Formulation ]
      ↓
[ Adaptive Depth & MoD Layer Pruning ]
      ↓
[ Hardware & Numerical Precision Selection (MPS / CPU / FP16 / INT8) ]
      ↓
[ Candidate Execution Engine (Measured / Simulated) ]
      ↓
[ Verification & Dynamic Stopping ]
      ↓
OpenAI-Compatible Response + Transparent Telemetry (x_aeon)
```

In addition to the core controller, AEON Phase 10 provides:
* **OpenAPI HTTP Server**: Drop-in OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`).
* **Web Dashboard**: Zero-dependency browser console (`/dashboard`) with live telemetry and empirical provenance tagging.
* **Benchmark Harness**: 20 canonical workloads and 11 baseline configurations with reproducible seed control.

---

## Core Capabilities

| Capability | Mechanism | Economic / Performance Impact |
| :--- | :--- | :--- |
| **VOI Model Routing** | Calibrated task-difficulty estimation | Avoids unnecessary large-model invocation |
| **Deterministic Tools** | Sandboxed Calculator, SymPy, Python sandbox | Exact zero-drift accuracy, sub-millisecond execution |
| **Multi-Tier Memory** | L1 exact KV, L4 semantic embedding cache | Zero-compute cache hits on repeated prompts |
| **Speculative Inference**| N-gram and lightweight draft model proposals | Reduces autoregressive latency in generative tasks |
| **Adaptive Reasoning** | Latent scratchpad vs. model escalation | Avoids multi-turn reasoning overhead on direct queries |
| **Adaptive Depth / MoD** | Intermediate classifier early exits (Phases 7 & 9)| **~37% FLOP reduction** via shallow layer termination |
| **Hardware & Precision** | Dynamic device placement (MPS/CPU/CUDA) & INT8 | Minimizes memory footprint and hardware energy |
| **Telemetry & Observability**| Bounded in-memory store, Prometheus metrics | Full empirical visibility with privacy by default |

---

## Phase 1–10 Roadmap & Current Status

| Phase | Milestone | Validation Status | Core Verification |
| :---: | :--- | :---: | :--- |
| **Phase 1** | Foundation & Architecture Assessment | **PASS** | Core abstractions, dual-backend simulation |
| **Phase 2** | VOI Controller & Task Classification | **PASS** | Utility calibration, cost-latency optimization |
| **Phase 3** | Multi-Tier Memory Hierarchy | **PASS** | L1–L4 caching, semantic safety guardrails |
| **Phase 4** | Tool-Aware Computation | **PASS** | Sandboxed execution, fallback mechanisms |
| **Phase 5** | Speculative Decoding & Token Efficiency| **PASS** | N-gram and draft verification |
| **Phase 6** | Adaptive Latent & Token-Efficient Reasoning| **PASS (WITH LIMITATIONS)** | Dynamic strategy selection |
| **Phase 7** | Adaptive Model Depth & Early Exits | **PASS (WITH LIMITATIONS)** | Layer truncation & Mixture-of-Depths |
| **Phase 8** | Hardware-Aware Precision & Quantization| **PASS (WITH LIMITATIONS)** | MPS/CPU allocation, memory safety guardrails |
| **Phase 9** | Full Benchmark Suite & Ablation Studies| **PASS (WITH LIMITATIONS)** | 20 workloads, 11 baselines, 10 ablations |
| **Phase 10**| OpenAPI Server & Web Dashboard | **PASS (WITH LIMITATIONS)** | OpenAI-compatible API, dashboard, hardening |

*Note on "PASS WITH LIMITATIONS": As detailed below, physical hardware execution is validated on Apple Silicon MPS and host CPU; datacenter GPU profiles (A100, H100) are analytical simulation baselines.*

---

## Scientific Claims & Empirical Results

### What the Evidence Supports (Phase 9 Validated Benchmark)

* **FLOP Reduction**: Full AEON achieves **~37% total FLOP reduction** (56.9 GFLOPs vs. 90.3 GFLOPs with depth disabled, and 2,192 GFLOPs for the fixed large baseline) across the canonical 20-workload test suite.
* **Causal Attribution**: Causal decomposition confirms that **Phase 7 adaptive depth early exit is the primary driver** of computational savings, accounting for over 98% of the FLOP delta.
* **Controller Overhead**: The multi-tier controller decision pipeline introduces a median wall-clock overhead of **0.201 ms** (mean 0.345 ms). Because execution savings exceed this overhead by over two orders of magnitude, controller overhead is fully amortized on the first request.
* **Hybrid Latency Reduction**: In the evaluated hybrid benchmark (simulated target execution + physically measured controller), AEON reduces end-to-end latency by up to **96.7%** relative to fixed large model execution.

### What AEON Does NOT Claim

* **Answer Quality**: The canonical benchmark did **NOT** demonstrate superior answer quality for Full AEON versus the direct fixed large baseline. Both configurations attained identical accuracy on supported tasks. **AEON should be understood as an adaptive efficiency and compute-orchestration system, not a system that increases model intelligence.**
* **Not Universal**: Results are benchmark-specific. Real-world latency and cost savings depend heavily on prompt length, model size, batch size, and local hardware.
* **Not Production GPU-Validated**: High-concurrency throughput on multi-node NVIDIA H100 clusters has not been physically measured.

---

## Benchmark Headline Results

The following validated results reflect the canonical Phase 9 evaluation (20 workloads across 12 taxonomy families):

| Configuration | Quality | Latency | Tokens | FLOPs | Cost | Quality/$ | Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full AEON (`B10_full_aeon`)** | 10.0% | 4.16 ms | 1116 | 56.9 G | $0.0000137 | 7,299.3 | `HYBRID` |
| **Direct Baseline (`B0_direct`)** | 15.0% | 126.19 ms | 3828 | 2192.4 G | $0.0000024 | 62,500.0 | `HYBRID` |
| **Fixed Large Baseline (`B3_fixed_large`)** | 15.0% | 126.17 ms | 3828 | 2192.4 G | $0.0000024 | 62,500.0 | `HYBRID` |
| **Depth OFF (`A7_depth_off`)** | 10.0% | 6.20 ms | 1098 | 90.3 G | $0.0000137 | 7,299.3 | `HYBRID` |
| **Hardware OFF (`A1_hardware_off`)** | 10.0% | 4.21 ms | 1116 | 57.8 G | $0.0000137 | 7,299.3 | `HYBRID` |
| **Speculation OFF (`A9_speculation_off`)** | 15.0% | 16.14 ms | 953 | 267.8 G | $0.0000139 | 10,791.4 | `HYBRID` |
| **Tool Routing OFF (`A11_tool_routing_off`)** | 0.0% | 9.77 ms | 1727 | 136.2 G | $0.0000002 | 0.0 | `HYBRID` |

---

## Empirical Provenance Standards

Every metric presented by AEON carries an explicit empirical provenance tag:

| Provenance Label | Definition | Examples in AEON |
| :--- | :--- | :--- |
| **`MEASURED`** | Directly measured on physical hardware via system timers or counters. | Controller overhead, host RAM headroom, MPS/CPU latency |
| **`SIMULATED`** | Generated via calibrated analytical or architectural hardware models. | Datacenter A100/H100 kernel latency, reference model output |
| **`HYBRID`** | Combines physical host measurements with calibrated simulation components. | End-to-end benchmark latency, pipeline throughput |
| **`ESTIMATED`** | Analytically calculated from cost tables, power ratings, or token counts. | Cost in USD ($), energy consumption in Joules (J) |
| **`DERIVED`** | Mathematically calculated from formal architectural formulas. | FLOP counts, parameter counts, Quality/$ efficiency score |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/aeon.git
cd aeon

# Install core runtime (lightweight, no mandatory heavy dependencies)
pip install -e .

# Optional: Install with full development and PyTorch dependencies
pip install -e ".[full]"
```

### 2. Verify Test Suite

```bash
# Run the complete test suite (386 passing tests)
pytest -q
```

### 3. Start the Inference Server

```bash
# Starts local-first OpenAPI server on 127.0.0.1:8000
aeon serve

# Start with an API key required for all requests
aeon serve --api-key my-secret-token

# Check server health
aeon server health
```

---

## API Usage

### `curl` Example

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

Response includes standard OpenAI fields plus non-intrusive `x_aeon` telemetry:

```json
{
  "id": "chatcmpl-aeon-1700000000000",
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
    "prompt_tokens": 10,
    "completion_tokens": 3,
    "total_tokens": 13
  },
  "x_aeon": {
    "selected_model": "aeon-small-1.5b",
    "selected_device": "mps",
    "selected_precision": "fp16",
    "selected_depth": 16,
    "tool_used": "calculator",
    "cache_hit": false,
    "latency_ms": 12.4,
    "provenance": "HYBRID"
  }
}
```

### Python OpenAI SDK Example

AEON works out-of-the-box with the official OpenAI Python library:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="aeon-local"  # Dummy key accepted when auth is disabled
)

# 1. Standard completion with policy controls
response = client.chat.completions.create(
    model="aeon-auto",
    messages=[
        {"role": "user", "content": "Explain Amdahl's Law concisely."}
    ],
    extra_body={
        "aeon": {
            "quality_target": 0.95,
            "latency_sla_ms": 500.0
        }
    }
)
print("Answer:", response.choices[0].message.content)

# 2. Server-Sent Events (SSE) streaming
stream = client.chat.completions.create(
    model="aeon-auto",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
```

---

## Web Dashboard

Access the built-in observable web console at:

```text
http://127.0.0.1:8000/dashboard
```

* Zero external JavaScript or CSS dependencies (vanilla HTML/CSS/JS).
* Real-time metrics: CPU/RAM/MPS detection, requests/sec, p50/p95/p99 latency percentiles.
* Visual breakdown of model selection, device allocation, precision, cache hit rate, and tool calls.
* Strict privacy: prompts and responses are **never** displayed or stored in dashboard telemetry.

---

## Benchmarking CLI

Execute validated Phase 1–9 benchmarks directly:

```bash
# List canonical workloads (20 items across 12 families)
aeon benchmark list

# Run full baseline suite
aeon benchmark run

# Compare baselines against Full AEON
aeon benchmark compare

# Run reproducibility check (verifies bit-for-bit determinism across runs)
aeon benchmark reproducibility

# Run static benchmark integrity scanner
aeon benchmark validate
```

---

## Limitations & Known Constraints

1. **Hardware Validation Scope**: Physical GPU execution is currently validated on Apple Silicon MPS and host CPU. Datacenter hardware profiles (NVIDIA A100 SXM4 80GB, NVIDIA H100 SXM5 80GB) are analytical simulation baselines.
2. **Default Simulation Backend**: AEON ships with calibrated analytical kernels to ensure reproducible, zero-cost experimentation without requiring gigabytes of model downloads. Adapters for local runtimes (`llama.cpp`, `vLLM`) can be plugged into the `ModelBackend` interface.
3. **Local-Scale Concurrency**: Load testing has been evaluated up to 8 concurrent in-flight streams on desktop hardware; multi-node cluster load has not been measured.

---

## Repository Structure

```text
aeon/
├── cli/                # Command line interface (`aeon serve`, `aeon benchmark`)
├── controller/         # VOI estimation, budget allocation, policy engine
├── decision/           # Calibrated decision models & task classification
├── depth/              # Adaptive depth early exits & Mixture-of-Depths router
├── hardware/           # Hardware capabilities, precision, memory preflight
├── inference/          # Dual backend abstraction (Simulation & Local)
├── memory/             # Multi-tier memory hierarchy (L1 exact to L4 semantic)
├── models/             # Provider-agnostic model registry & capability profiles
├── reasoning/          # Adaptive latent & explicit reasoning controller
├── routing/            # Cascade and task classification routing
├── server/             # OpenAPI HTTP server, SSE streaming, auth, rate limiting
│   ├── adapters/       # OpenAI Chat Completions compatibility adapter
│   ├── dashboard/      # Vanilla HTML/CSS/JS web dashboard console
│   └── telemetry/      # Bounded ring-buffer telemetry store & Prometheus exporter
├── speculation/        # N-gram & draft model speculative decoding
├── tools/              # Sandboxed deterministic tools (Calculator, SymPy, Python)
artifacts/              # Validated benchmark manifests and run output
docs/                   # Architecture manuals, API reference, phase reports
examples/               # Ready-to-run client scripts (curl, Python SDK)
tests/                  # 386 comprehensive unit, integration, and contract tests
```

---

## Development & Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run complete test suite (Phases 1–10)
pytest -q

# Run Phase 10 API & Server tests specifically
pytest tests/test_phase10.py -v
```

---

## Contributing

Contributions, feedback, and issue reports are welcome! Please consult [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

> **Core Research Rule**: All benchmark contributions and performance claims must preserve empirical provenance and must not represent simulated or estimated measurements as physical hardware measurements.

---

## Security

For security vulnerability disclosure, please review our [Security Policy](SECURITY.md).

---

## License

License: Apache-2.0

AEON is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use or reference AEON in your research or engineering projects, please cite:

```bibtex
@software{aeon2026,
  title = {AEON: Adaptive Economically-Optimized Network for AI Inference Orchestration},
  author = {AEON Contributors},
  year = {2026},
  url = {https://github.com/your-org/aeon},
  version = {0.1.0}
}
```

