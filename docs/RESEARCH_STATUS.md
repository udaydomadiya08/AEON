# AEON Research Status & Scientific Boundary Matrix

This document provides a rigorous, transparent classification of AEON's empirical and theoretical claims as of version 0.1.0.

In accordance with our research integrity principles, claims are divided into three distinct categories:
1. **ESTABLISHED**: Experimentally verified with reproducible code, automated tests, and consistent benchmark measurements.
2. **SUPPORTED BUT LIMITED**: Empirically demonstrated under specific constrained conditions or hybrid evaluation frameworks, with known limitations.
3. **NOT YET ESTABLISHED**: Hypothesized or modeled theoretically, but not yet experimentally validated on physical production infrastructure.

---

## 1. ESTABLISHED Claims

* **Computational FLOP Reduction via Adaptive Depth**:
  * Experimentally established **~37% total FLOP reduction** across the canonical 20-workload benchmark suite.
  * Causal $2 \times 2$ factorial ablation proves that Phase 7 early exit layer truncation is the primary causal driver (accounting for >98% of the isolated FLOP reduction).
* **Deterministic Tool Route Efficiency**:
  * Arithmetic and mathematical expressions (e.g. `17 * 23`) route to lightweight sandboxed tools with exact zero-drift outputs and sub-millisecond execution times, completely avoiding neural parameter evaluation.
* **Controller Overhead Amortization**:
  * The multi-tier controller decision pipeline introduces a median wall-clock overhead of **0.201 ms** (mean 0.345 ms).
  * This overhead is amortized on the first inference request against typical baseline model latencies (>100 ms).
* **Multi-Tier Memory Cache Hits**:
  * Identical and semantic cache queries resolve with zero neural inference compute, reducing execution latency from ~126 ms to sub-millisecond retrieval.
* **Deterministic Reproducibility**:
  * Benchmark runs executed under a fixed pseudorandom seed (`seed=42`) yield bit-for-bit identical metric results across repeated independent executions.
* **OpenAPI Contract & API Compatibility**:
  * Full HTTP compatibility with the OpenAI Chat Completions specification (`/v1/chat/completions` and `/v1/models`), supporting both synchronous requests and Server-Sent Events (SSE) streaming.

---

## 2. SUPPORTED BUT LIMITED Claims

* **Hybrid End-to-End Latency Reduction**:
  * Up to **96.7% latency reduction** is observed in the canonical benchmark.
  * *Limitation*: The benchmark operates in a hybrid regime: controller decisions, tool execution, and local dispatch are physically measured on Apple Silicon MPS and host CPU, while target model forward passes are simulated using calibrated analytical timing curves. Real latency reductions with heavy local model weights depend on model size and hardware memory bandwidth.
* **Hardware-Aware Acceleration on Apple Silicon**:
  * Device placement and memory headroom detection are functional on Apple Silicon Metal Performance Shaders (MPS) and CPU.
  * *Limitation*: Validated on consumer M-series hardware (8 GB - 64 GB unified memory); not yet evaluated on enterprise unified memory nodes (e.g. 128 GB+ M2/M3 Max/Ultra).
* **Local Concurrency & Backpressure**:
  * Concurrency limiting and backpressure queue rejection (HTTP 503 on saturation) are verified locally under concurrent dispatch (up to 8 in-flight streams).
  * *Limitation*: Evaluated on local host environments, not multi-tenant cloud clusters under thousands of simultaneous requests.

---

## 3. NOT YET ESTABLISHED Claims

The following claims are **explicitly NOT established** in AEON v0.1.0:

* **Physical NVIDIA A100 / H100 GPU Validation**:
  * Datacenter GPU execution profiles (A100 SXM4 80GB, H100 SXM5 80GB) are currently calibrated analytical simulations. Physical multi-GPU tensor-parallel execution has not been measured.
* **Superior Output Quality / Increased Intelligence**:
  * The canonical benchmark did **NOT** demonstrate higher answer quality for Full AEON compared to the fixed large baseline. Both configurations achieved identical high task accuracy. AEON is an adaptive compute efficiency and cost orchestration system, not a mechanism that increases foundational model intelligence.
* **Universal Cost & Latency Reductions**:
  * Compute and latency savings are workload- and prompt-dependent. For dense, ambiguous multi-turn tasks that cannot exit early or utilize tools, compute savings will be minimal.
* **Native 70B Model FP8 / INT4 Weight Execution**:
  * Real 70B parameter weight matrices have not been loaded or evaluated on native FP8 tensor cores in this development cycle.
* **High-Throughput Production Cluster Serving**:
  * Continuous multi-worker production serving across distributed Kubernetes clusters under high network latency and multi-tenant traffic has not yet been stress-tested.
