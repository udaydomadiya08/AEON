"""
AEON Phase 9 — Comparative Analysis & Multi-Objective Pareto Frontier.

Computes comparative metric differentials and identifies Pareto-optimal
points across multiple conflicting objectives:
Maximize Quality, Minimize Latency, Cost, Tokens, FLOPs, Energy.
"""

from typing import List, Dict, Any, Tuple
from aeon.benchmarks.types import ComparisonResult, BenchmarkResult


def compute_comparison(candidate: BenchmarkResult, baseline: BenchmarkResult) -> ComparisonResult:
    """Calculate comparative gains and reduction percentages."""
    d_qual = candidate.quality - baseline.quality
    d_lat = candidate.mean_latency_ms - baseline.mean_latency_ms
    d_cost = candidate.mean_cost_usd - baseline.mean_cost_usd
    d_tok = candidate.total_tokens - baseline.total_tokens
    d_flops = candidate.mean_flops - baseline.mean_flops
    d_energy = candidate.mean_energy_joules - baseline.mean_energy_joules
    d_qpd = candidate.quality_per_dollar - baseline.quality_per_dollar

    lat_red = (1.0 - candidate.mean_latency_ms / baseline.mean_latency_ms) * 100.0 if baseline.mean_latency_ms > 0 else 0.0
    cost_red = (1.0 - candidate.mean_cost_usd / baseline.mean_cost_usd) * 100.0 if baseline.mean_cost_usd > 0 else 0.0
    tok_red = (1.0 - candidate.total_tokens / baseline.total_tokens) * 100.0 if baseline.total_tokens > 0 else 0.0

    # Pareto superiority check: at least as good in all objectives, strictly better in at least one
    objectives_candidate = (
        candidate.quality,
        -candidate.mean_latency_ms,
        -candidate.mean_cost_usd,
        -candidate.mean_energy_joules
    )
    objectives_baseline = (
        baseline.quality,
        -baseline.mean_latency_ms,
        -baseline.mean_cost_usd,
        -baseline.mean_energy_joules
    )

    as_good = all(c >= b for c, b in zip(objectives_candidate, objectives_baseline))
    strictly_better = any(c > b for c, b in zip(objectives_candidate, objectives_baseline))
    pareto_sup = as_good and strictly_better

    return ComparisonResult(
        baseline_name=baseline.condition_name,
        candidate_name=candidate.condition_name,
        delta_quality=round(d_qual, 4),
        delta_latency=round(d_lat, 2),
        delta_cost=round(d_cost, 7),
        delta_tokens=d_tok,
        delta_flops=round(d_flops, 2),
        delta_energy=round(d_energy, 4),
        delta_quality_per_dollar=round(d_qpd, 2),
        latency_reduction_pct=round(lat_red, 2),
        cost_reduction_pct=round(cost_red, 2),
        token_reduction_pct=round(tok_red, 2),
        is_pareto_superior=pareto_sup
    )


def compute_pareto_frontier(candidates: List[BenchmarkResult]) -> Tuple[List[BenchmarkResult], List[BenchmarkResult]]:
    """
    Identify non-dominated points in the multi-objective space:
    Maximize Quality, Minimize Latency, Cost, Energy.
    Returns (pareto_optimal_list, dominated_list).
    """
    pareto_optimal: List[BenchmarkResult] = []
    dominated: List[BenchmarkResult] = []

    for i, cand in enumerate(candidates):
        is_dominated = False
        c_vec = (
            cand.quality,
            -cand.mean_latency_ms,
            -cand.mean_cost_usd,
            -cand.mean_energy_joules
        )

        for j, other in enumerate(candidates):
            if i == j:
                continue
            o_vec = (
                other.quality,
                -other.mean_latency_ms,
                -other.mean_cost_usd,
                -other.mean_energy_joules
            )

            # other dominates cand if other >= cand in all and > in at least one
            if all(o >= c for o, c in zip(o_vec, c_vec)) and any(o > c for o, c in zip(o_vec, c_vec)):
                is_dominated = True
                break

        if is_dominated:
            dominated.append(cand)
        else:
            pareto_optimal.append(cand)

    return pareto_optimal, dominated
