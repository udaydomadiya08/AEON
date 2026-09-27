# AEON PHASE 5: SPECULATIVE & TOKEN-EFFICIENT INFERENCE
## Architectural Documentation & Quickstart Guide

AEON Phase 5 integrates speculative decoding and token-efficient generation as an **economic computational decision** within AEON's Compute Controller and Value-of-Information (VOI) framework.

### Central Research Axiom
> **"Never spend speculative computation when the drafting and verification overhead exceeds the expected accepted-token benefit."**

---

## 1. Key Features

- **Heterogeneous Speculation Strategies**:
  - `STANDARD`: Standard autoregressive decoding baseline ($\mathcal{O}(2NT)$).
  - `DRAFT_MODEL`: Calibrated speculative drafting with compatible low-parameter models (e.g. `aeon-draft-0.5b`, `aeon-small-1.5b`).
  - `NGRAM`: Sub-millisecond prompt and context n-gram frequency speculation.
  - `MULTI_TOKEN`: Multi-token prediction primitive representation.
- **Online Adaptive Acceptance Tracking**:
  - Exponential moving average (EMA) tracked across global, per-strategy, per-model, and per-draft-length dimensions.
  - Automatic deterioration detection (`is_deteriorating()`) detecting decay across recent rounds and triggering safe fallback.
- **Dynamic Draft Length Optimization**:
  - Evaluates lengths $\gamma \in \{1, 2, 4, 6, 8, 16\}$ against length-decay curves to maximize expected net utility.
- **First-Class Model Registry Integration**:
  - Draft models are registered in `ModelRegistry` with compatibility verification (`is_compatible()`).
- **Parallel Token Verification Engine**:
  - Sequential prefix verifier detecting divergence and emitting replacement tokens.
- **Hardware Honesty Guarantee**:
  - Clearly distinguishes `SIMULATED_SPECULATIVE_DECODING` from `REAL_SPECULATIVE_DECODING`.

---

## 2. CLI Quick Reference

```bash
# 1. Run inference with speculation enabled and explainable telemetry trace
aeon run "Write a Python function to compute the Fibonacci sequence." --speculate --trace

# 2. Run full Phase 5 Benchmark Suite (Standard vs Always-On vs AEON Adaptive)
aeon benchmark --phase 5

# 3. Run Component Ablation Studies (isolating draft length, learning, n-gram)
aeon benchmark --speculation-ablation

# 4. Run Theoretical Economic Break-Even Analysis across models and lengths
aeon benchmark --speculation-break-even

# 5. Run Generation Workload Breakdown across 8 modalities
aeon benchmark --speculation-workload

# 6. Inspect Active Speculation Statistics
aeon speculation stats

# 7. Inspect Active Speculation Configuration
aeon speculation config

# 8. Compare Speculation Strategies Across Workloads
aeon speculation compare

# 9. Run Break-Even Matrix
aeon speculation break-even
```

---

## 3. Python API Quickstart

```python
from aeon import AEON, InferenceRequest, SpeculationStrategy

# Initialize AEON with speculation and memory enabled
aeon = AEON(policy="voi", memory=True, speculation=True)

# Run inference
result = aeon.run("Explain how transformer attention operates step-by-step.", speculation=True)

print("Answer:", result.answer)
print("Strategy Chosen:", result.trace.speculation_strategy)
print("Realized Speedup:", result.trace.realized_speedup)
print("Acceptance Rate:", result.trace.acceptance_rate)
print("Execution Mode:", result.trace.speculation_mode)
```

---

## 4. Configuration Options

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `enabled` | bool | `True` | Master toggle for speculation |
| `strategies` | list | `[standard, draft_model, ngram, multi_token]` | Active candidate strategies |
| `candidate_draft_lengths`| list | `[1, 2, 4, 6, 8, 16]` | Evaluated speculation depths |
| `default_draft_length` | int | `4` | Starting speculation depth |
| `min_acceptance_rate` | float | `0.50` | Threshold required to maintain speculation |
| `fallback_threshold` | float | `0.35` | Deterioration threshold triggering fallback |
| `learning_rate` | float | `0.25` | Online EMA smoothing factor ($\alpha$) |
| `max_speculation_rounds`| int | `50` | Max rounds before termination |

---

## 5. Test Suite Verification

Run the full regression test suite (Phases 1–5):
```bash
python3 -m pytest -v
```
**Current Status**: 110/110 tests passing (100% pass rate).
