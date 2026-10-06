# AEON Web Dashboard Manual

## Overview

The AEON Web Dashboard provides real-time, zero-dependency visibility into AEON's adaptive compute orchestrator. It is hosted directly on the AEON server at:

```text
http://127.0.0.1:8000/dashboard
```

---

## 1. Architecture & Design Principles

- **Zero Heavy Frontend Dependencies**: Built entirely with semantic HTML5, modern CSS3 variables and glassmorphism styling, and vanilla JavaScript. No React, Vue, webpack, or npm builds are required.
- **Local-First & Lightweight**: The entire dashboard footprint is under 30KB, loads instantaneously, and consumes negligible system resources.
- **Configurable Polling**: Background polling defaults to 5 seconds and can be adjusted via the UI (1s, 5s, 10s, 30s, or Paused) to prevent unnecessary polling overhead.
- **Strict Privacy**: The dashboard never displays raw user prompts, conversation content, or sensitive API keys. Only metadata (latencies, token counts, model routing, costs) is shown.

---

## 2. Dashboard Sections & Metrics

### 1. System Telemetry
- **Host Accelerator**: Displays detected hardware backend (e.g. Apple Silicon MPS, NVIDIA CUDA, CPU).
- **Physical Host**: Platform architecture (e.g. `arm64`, `x86_64`) and Python runtime version.
- **Memory Footprint**: Available system RAM headroom and memory guard constraints.
- **AEON Uptime**: Process uptime since server startup.

### 2. Request & Latency Analytics
- **Total Invocations**: Total HTTP requests processed.
- **Success / Failure Count**: Status code breakdown (2xx vs 4xx/5xx).
- **Latency Distribution**:
  - Average Latency (ms)
  - Median (p50) Latency (ms)
  - 95th Percentile (p95) Latency (ms)
  - 99th Percentile (p99) Latency (ms)

### 3. AEON Dynamic Orchestration Decisions
- **Model Distribution**: Breakdown of routed model profiles (`aeon-auto`, `aeon-small-1.5b`, `aeon-medium-7b`, `aeon-large-70b`).
- **Precision Distribution**: Dynamically selected numerical precision (`FP32`, `FP16`, `BF16`, `INT8`, `FP8`).
- **Device Placement**: Workload allocation across `mps`, `cuda`, and `cpu`.
- **Cache Hit Rate**: Percentage of queries resolved via Phase 3 exact or semantic memory.
- **Speculation Rate**: Percentage of requests accelerating generation with speculative draft decoding.
- **Deterministic Tool Routing**: Volume and distribution of tool calls (Calculator, Python sandbox, SymPy, Retrieval).
- **Adaptive Reasoning**: Strategy selection breakdown (Direct, Latent, Explicit, Escalation).
- **Execution Depth**: Early exit layer distribution and Mixture-of-Depths router bypass rate.

### 4. Economic Telemetry
- **Estimated Monetary Cost ($)**: Total and per-request expenditure based on model pricing profiles.
- **Estimated Energy Consumption (J)**: Hardware energy usage calculated from TDP and device utilization.
- **Quality per Dollar ($)**: Economic efficiency score measuring verified output quality per unit resource cost.
- **Token Accounting**: Cumulative prompt, completion, and total token throughput.

---

## 3. Explicit Provenance Tagging

In accordance with AEON's scientific integrity standards, every metric card explicitly discloses its empirical provenance:

| Metric Category | Provenance Tag | Meaning |
| :--- | :--- | :--- |
| **Local Latency** | `MEASURED` / `HYBRID` | Measured via high-resolution monotonic timer on the local host. |
| **System Device** | `MEASURED` | Verified directly against PyTorch device backends (`torch.backends.mps` or `torch.cuda`). |
| **Cost** | `ESTIMATED` | Calculated analytically from model token consumption and profile rate tables. |
| **Energy** | `ESTIMATED` | Derived from hardware TDP power models and execution duration. |
| **FLOPs** | `DERIVED` | Analytically computed from model parameter counts and active execution depth. |

Simulated datacenter devices (e.g. A100 SXM4 80GB, H100 SXM5 80GB) are explicitly labeled with `SIMULATED` provenance.

---

## 4. Dashboard APIs

The dashboard UI consumes two lightweight JSON endpoints:

### `GET /dashboard/api/summary`
Returns the aggregated system state, latency percentiles, model routing distributions, economics, and provenance tags.

### `GET /dashboard/api/recent`
Returns the 50 most recent completed request records with prompt text redacted for privacy.
