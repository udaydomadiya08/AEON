"""
AEON Phase 9 — Statistical Repetitions & Saturation Detector.

Orchestrates multi-trial repetitions over workloads and identifies benchmark
saturation anomalies (e.g. 100% uniform quality or frozen cost).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from aeon.benchmarks.types import BenchmarkTrial, BenchmarkResult, StatisticalSummary
from aeon.benchmarks.statistics import compute_statistical_summary


def detect_saturation_warnings(trials: List[BenchmarkTrial]) -> List[str]:
    """Detect suspicious uniformity or saturation across trials."""
    warnings = []
    if not trials:
        return warnings

    # Check 1: 100% uniform quality across all trials
    qualities = [1.0 if t.correct else 0.0 for t in trials]
    if len(trials) >= 5 and all(q == 1.0 for q in qualities):
        warnings.append("SATURATION_WARNING: 100.0% quality observed uniformly across all trials; check task difficulty diversity.")

    # Check 2: Identical cost across all trials
    costs = [t.cost_usd for t in trials]
    if len(trials) >= 5 and len(set(costs)) == 1:
        warnings.append("SATURATION_WARNING: Identical cost_usd across all trials; verify cost scaling model.")

    # Check 3: Identical latency across all trials
    lats = [t.latency_ms for t in trials]
    if len(trials) >= 5 and len(set(lats)) == 1:
        warnings.append("SATURATION_WARNING: Identical latency across all trials; verify latency modeling.")

    return warnings


def aggregate_trials_to_result(
    condition_name: str,
    trials: List[BenchmarkTrial]
) -> BenchmarkResult:
    """Aggregate a collection of trials into an auditable BenchmarkResult."""
    n = len(trials)
    if n == 0:
        raise ValueError("Cannot aggregate empty trials list")

    correct_count = sum(1 for t in trials if t.correct)
    quality = correct_count / n

    latencies = [t.latency_ms for t in trials]
    costs = [t.cost_usd for t in trials]
    tokens = [t.tokens for t in trials]
    flops = [t.flops for t in trials]
    memories = [t.memory_mb for t in trials]
    energies = [t.energy_joules for t in trials]
    utilities = [t.utility for t in trials]
    depths = [t.depth_layers for t in trials]
    overheads = [t.controller_overhead_ms for t in trials]

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    mean_cost = float(np.mean(costs))
    tot_tokens = int(sum(tokens))
    mean_flops = float(np.mean(flops))
    mean_mem = float(np.mean(memories))
    mean_energy = float(np.mean(energies))
    mean_utility = float(np.mean(utilities))
    mean_depth = float(np.mean(depths))
    mean_overhead = float(np.mean(overheads))

    rounded_quality = round(quality, 4)
    rounded_cost = round(mean_cost, 7)
    safe_cost = rounded_cost if rounded_cost > 1e-9 else 1e-6
    safe_lat_s = (mean_lat / 1000.0) if mean_lat > 1e-6 else 0.001
    safe_tflops = (mean_flops / 1e12) if mean_flops > 1e6 else 0.001

    quality_per_dollar = round(rounded_quality / safe_cost, 2)
    quality_per_second = round(rounded_quality / safe_lat_s, 2)
    quality_per_tflops = round(rounded_quality / safe_tflops, 2)
    overhead_pct = round((mean_overhead / max(0.001, mean_lat)) * 100.0, 2)

    # Statistical summaries
    stats = {
        "latency": compute_statistical_summary(latencies),
        "cost": compute_statistical_summary(costs),
        "flops": compute_statistical_summary(flops),
        "energy": compute_statistical_summary(energies),
        "overhead": compute_statistical_summary(overheads)
    }

    saturation_warnings = detect_saturation_warnings(trials)

    # Modal device, precision, quantization
    device = trials[-1].device
    prec = trials[-1].precision
    quant = trials[-1].quantization

    return BenchmarkResult(
        condition_name=condition_name,
        num_requests=n,
        quality=round(quality, 4),
        mean_latency_ms=round(mean_lat, 2),
        median_latency_ms=round(median_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        mean_cost_usd=round(mean_cost, 7),
        total_tokens=tot_tokens,
        mean_flops=round(mean_flops, 1),
        mean_memory_mb=round(mean_mem, 1),
        mean_energy_joules=round(mean_energy, 4),
        mean_utility=round(mean_utility, 4),
        quality_per_dollar=quality_per_dollar,
        quality_per_second=quality_per_second,
        quality_per_tflops=quality_per_tflops,
        mean_depth=round(mean_depth, 1),
        selected_device=device,
        selected_precision=prec,
        quantization=quant,
        controller_overhead_ms=round(mean_overhead, 4),
        overhead_pct=overhead_pct,
        break_even_requests=1.0,
        statistical_metrics=stats,
        saturation_warnings=saturation_warnings
    )
