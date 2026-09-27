"""
Benchmark Metrics and Comparative Efficiency Calculator.

Implements formal efficiency evaluation:
Efficiency = Capability / (Latency * Cost * FLOPs * Energy)
and individual Quality/Resource ratios.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
from aeon.types import Telemetry


class AggregateMetrics(BaseModel):
    name: str
    num_requests: int
    accuracy: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    reasoning_tokens: int
    total_compute_flops: float
    mean_compute_flops: float
    total_cost_usd: float
    mean_cost_usd: float
    total_energy_joules: float
    escalation_rate: float
    
    # Efficiency Ratios
    quality_per_dollar: float  # Accuracy / Total Cost USD
    quality_per_second: float  # Accuracy / Mean Latency (s)
    quality_per_tflops: float  # Accuracy / Mean TFLOPs


def calculate_aggregate_metrics(name: str, traces: List[Telemetry]) -> AggregateMetrics:
    """Calculate aggregate efficiency metrics from execution traces."""
    n = len(traces)
    if n == 0:
        raise ValueError("Cannot calculate metrics on empty traces")

    latencies = [t.latency_ms for t in traces]
    total_tokens = sum(t.input_tokens + t.output_tokens for t in traces)
    reasoning_tokens = sum(t.reasoning_tokens for t in traces)
    total_flops = sum(t.compute_estimate for t in traces)
    total_cost = sum(t.estimated_cost for t in traces)
    total_energy = sum(t.estimated_energy for t in traces)
    escalations = sum(1 for t in traces if t.escalated)

    # Accuracy calculation based on ground truth evaluation or confidence threshold
    correct_count = sum(1 for t in traces if t.correct is True or (t.correct is None and t.confidence >= 0.70))
    accuracy = correct_count / n

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    mean_flops = total_flops / n
    mean_cost = total_cost / n

    mean_lat_s = mean_lat / 1000.0 if mean_lat > 0 else 0.001
    mean_tflops = mean_flops / 1e12 if mean_flops > 0 else 0.001
    safe_cost = total_cost if total_cost > 0 else 1e-6

    quality_per_dollar = accuracy / safe_cost
    quality_per_second = accuracy / mean_lat_s
    quality_per_tflops = accuracy / mean_tflops

    return AggregateMetrics(
        name=name,
        num_requests=n,
        accuracy=round(accuracy, 4),
        mean_latency_ms=round(mean_lat, 2),
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        total_compute_flops=total_flops,
        mean_compute_flops=mean_flops,
        total_cost_usd=round(total_cost, 6),
        mean_cost_usd=round(mean_cost, 6),
        total_energy_joules=round(total_energy, 4),
        escalation_rate=round(escalations / n, 4),
        quality_per_dollar=round(quality_per_dollar, 2),
        quality_per_second=round(quality_per_second, 2),
        quality_per_tflops=round(quality_per_tflops, 4),
    )
