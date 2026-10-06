# AEON v0.1.0 Release Notes
## First Public Engineering Release

We are pleased to introduce the first public engineering release of **AEON (Adaptive Economically-Optimized Network)**.

AEON is an adaptive compute orchestration layer designed to dynamically route inference across available models, deterministic tools, multi-level memory, adaptive reasoning strategies, execution depth (early exit), numerical precision, and target hardware accelerators.

> **Status Notice**: AEON v0.1.0 is an initial engineering and research release. While the server, API contract, and orchestration controllers are hardened with 386 passing automated tests, physical GPU execution has been evaluated on Apple Silicon MPS and host CPU; datacenter GPU profiles (NVIDIA A100, H100) are currently analytical simulation baselines.

---

## 1. Release Highlights

* **OpenAPI-3.1 Compatible Inference Server**:
  * Drop-in replacement for OpenAI Chat Completions (`POST /v1/chat/completions`) and models (`GET /v1/models`).
  * Supports both synchronous responses and Server-Sent Events (SSE) streaming with `data: [DONE]`.
  * Non-intrusive `x_aeon` telemetry reporting selected model, execution depth, device, precision, and tool routing without breaking standard OpenAI client SDKs.
* **Zero-Dependency Web Dashboard**:
  * Interactive browser console hosted at `http://127.0.0.1:8000/dashboard`.
  * Real-time metrics: CPU/RAM/MPS system status, request volume, p50/p95/p99 latency percentiles, and decision breakdown.
  * Explicit provenance tagging (`MEASURED`, `SIMULATED`, `HYBRID`, `ESTIMATED`, `DERIVED`) on every card.
* **Multi-Tier Adaptive Compute Engine**:
  * **VOI Controller (Phase 2)**: Calibrated difficulty scoring and utility optimization.
  * **Multi-Tier Memory (Phase 3)**: Exact L1 KV cache and semantic L4 caching for zero-compute hits.
  * **Deterministic Tools (Phase 4)**: Sandboxed Calculator, SymPy, and Python sandbox.
  * **Speculative Decoding (Phase 5)**: N-gram and draft model proposals.
  * **Adaptive Reasoning (Phase 6)**: Dynamic direct, latent, and escalation strategies.
  * **Adaptive Depth Early Exits (Phase 7 & 9)**: Intermediate classifier exits delivering **~37% FLOP reduction** in benchmark evaluation.
  * **Hardware & Quantization (Phase 8)**: Dynamic device placement (MPS/CPU) and numerical precision selection.
* **Production Hardening & Safety**:
  * Localhost default (`127.0.0.1`) with prominent public binding warnings if bound to `0.0.0.0`.
  * Constant-time API key verification (`hmac.compare_digest`).
  * Defensive HTTP security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`).
  * In-memory token-bucket rate limiter and concurrency backpressure queue.
  * Memory safety preflight verification preventing host Out-of-Memory crashes.
  * Bounded ring-buffer telemetry store with age pruning and privacy-by-default secret scrubbing.

---

## 2. Architecture & Pipeline

```text
HTTP Request
     ↓
Authentication & Request ID Normalization
     ↓
Pydantic Validation & Size Guardrails
     ↓
AEON Task Classification & VOI Estimation
     ↓
Memory Hierarchy Lookup (L1 Exact / L4 Semantic)
     ↓
Deterministic Tool Routing (Calculator / SymPy / Python)
     ↓
Adaptive Reasoning Selection (Direct / Latent / Escalation)
     ↓
Speculative Draft Formulation
     ↓
Adaptive Depth & MoD Early Exit Truncation
     ↓
Hardware & Precision Selection (MPS / CPU / FP16 / INT8)
     ↓
Candidate Execution Engine
     ↓
Verification & Dynamic Stopping
     ↓
OpenAI-Compatible Response + Transparent Telemetry (x_aeon)
```

---

## 3. Installation & Getting Started

```bash
# Clone the repository
git clone https://github.com/your-org/aeon.git
cd aeon

# Install core package
pip install -e .

# Run test suite
pytest -q

# Start server
aeon serve --host 127.0.0.1 --port 8000
```

---

## 4. Benchmark Reproducibility & Environment

The validated benchmark results in v0.1.0 were executed under the following canonical environment:
* **Host Platform**: Darwin 24.x (Apple Silicon arm64, macOS 15.4)
* **CPU / Memory**: 8 cores, 8 GB unified memory
* **Accelerators**: Apple Silicon MPS (Metal Performance Shaders active); CUDA unavailable
* **Python Runtime**: Python 3.9.18
* **Benchmark Scope**: 20 canonical workloads across 12 taxonomy families evaluated over 11 baseline configurations and 13 ablations.
* **Seed Control**: Fixed seed (`seed=42`) yielding bit-for-bit reproducible runs across independent executions.

---

## 5. Known Limitations

1. **Analytical Datacenter Profiles**: NVIDIA A100 SXM4 and H100 SXM5 execution profiles are calibrated analytical simulations.
2. **Simulation Backend**: By default, AEON uses simulated neural execution to avoid requiring massive model weights to be downloaded. External runtimes (`llama.cpp`, `vLLM`) can be integrated via the `BaseInferenceBackend` interface.
3. **Answer Quality**: The benchmark did not demonstrate superior answer quality for Full AEON versus fixed large model baselines. AEON optimizes compute, latency, and cost efficiency, not foundational model intelligence.
