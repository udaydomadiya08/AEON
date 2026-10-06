# Changelog

All notable changes to the AEON project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-06

### Initial Public Engineering Release

#### Phase 1: Foundation & Architecture Assessment
* Established core AEON abstractions: task classification, difficulty estimation, budget allocation, policy engine, and decision models.
* Created dual-backend inference abstraction supporting `SimulationBackend` for deterministic, zero-cost benchmarking and `LocalBackend` for device execution.
* Established initial testing framework and research verification baseline.

#### Phase 2: Value-of-Information (VOI) Controller
* Implemented calibrated task difficulty scoring and utility estimation.
* Added multi-objective optimization balancing output quality, latency SLAs, monetary cost, and compute FLOPs.
* Introduced cascade routing with early stopping confidence thresholds.

#### Phase 3: Multi-Tier Memory Hierarchy
* Implemented 4-tier memory architecture: L1 exact key-value cache, L2 prefix cache, L3 intermediate procedural cache, and L4 semantic embedding cache.
* Integrated semantic similarity safety guardrails and admission/eviction policies.
* Delivered zero-compute cache hit resolution for repeated queries.

#### Phase 4: Deterministic Tool-Aware Computation
* Implemented sandboxed computational tools: Calculator, SymPy symbolic solver, and Python execution sandbox.
* Added deterministic tool candidate generation and planner.
* Integrated tool verification and graceful fallback to model paths upon tool exceptions.

#### Phase 5: Speculative Decoding & Token Efficiency
* Added speculative decoding controller supporting n-gram matching and lightweight draft model verification.
* Implemented draft length optimization and dynamic acceptance rate estimation.
* Added token efficiency metrics and speculative speedup tracking.

#### Phase 6: Adaptive Latent & Token-Efficient Reasoning
* Implemented adaptive reasoning controller supporting direct execution, latent scratchpad planning, explicit multi-step reasoning, and model escalation.
* Added complexity-based reasoning strategy selection.
* Validated dynamic reasoning transitions across varied problem domains.

#### Phase 7: Adaptive Model Depth & Mixture-of-Depths
* Implemented early exit layer truncation using intermediate calibrated classifiers.
* Added Mixture-of-Depths (MoD) routing to skip non-essential transformer layers.
* Established adaptive depth as the primary driver of FLOP reduction.

#### Phase 8: Hardware-Aware Precision & Quantization
* Implemented hardware detection for Apple Silicon (Metal Performance Shaders / MPS), host CPU, and CUDA.
* Added dynamic precision routing across FP32, FP16, BF16, INT8, and FP8.
* Integrated memory safety preflight checks to prevent host Out-of-Memory crashes.
* Explicitly established empirical provenance tagging (`MEASURED` vs. `SIMULATED`).

#### Phase 9: Full Benchmark Suite & Scientific Ablations
* Created canonical 20-workload benchmark suite across 12 problem taxonomy families.
* Implemented 11 baseline configurations and 13 ablation variants.
* Validated ~37% FLOP reduction and ~96.7% hybrid latency reduction.
* Performed causal decomposition proving Phase 7 adaptive depth is the primary causal driver of computational savings.
* Added bit-for-bit reproducibility verification with fixed seed control (`seed=42`).
* Implemented static AST integrity scanner to detect shortcuts or hardcoded evaluation constants.

#### Phase 10: OpenAPI Server & Observable Web Dashboard
* Built OpenAPI-3.1 compliant HTTP server exposing standard OpenAI-compatible endpoints (`/v1/chat/completions` and `/v1/models`).
* Supported synchronous completions and Server-Sent Events (SSE) streaming with `data: [DONE]`.
* Built zero-dependency dark-mode web dashboard (`/dashboard`) in pure HTML5/CSS3/JS.
* Implemented bounded in-memory telemetry store with age pruning and Prometheus exporter (`/metrics`).
* Added production hardening: constant-time API key auth (`hmac.compare_digest`), defensive HTTP security headers, in-memory rate limiting, and backpressure queueing.
* Integrated CLI management commands: `aeon serve`, `aeon server health`, `aeon server status`.
* Exported verified OpenAPI specification (`docs/openapi.json`).
