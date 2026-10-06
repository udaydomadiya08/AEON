"""
AEON Phase 10 — Telemetry Aggregator & Metrics Exporter.

Aggregates operational traces into summary statistics, latency percentiles,
routing distributions, economic summaries, and Prometheus exposition formatting.
"""

from typing import Dict, Any, List
import numpy as np
from aeon.server.telemetry.store import TelemetryStore


class TelemetryAggregator:
    """Aggregates raw telemetry events into structured analytical summaries."""

    def __init__(self, store: TelemetryStore):
        self.store = store

    async def get_summary(self) -> Dict[str, Any]:
        """Compute full statistical summary across all current records."""
        records = await self.store.get_all_records()
        n = len(records)

        if n == 0:
            return self._empty_summary()

        successful = [r for r in records if r.get("status_code", 200) < 400]
        failed = [r for r in records if r.get("status_code", 200) >= 400]
        timeouts = [r for r in records if r.get("status_code") in [408, 504]]

        # Latencies
        lats = [float(r["latency_ms"]) for r in successful if "latency_ms" in r]
        costs = [float(r["cost_usd"]) for r in successful if "cost_usd" in r]
        energies = [float(r["energy_joules"]) for r in successful if "energy_joules" in r]
        overheads = [float(r["controller_overhead_ms"]) for r in successful if "controller_overhead_ms" in r]
        depths = [float(r["depth"]) for r in successful if "depth" in r]

        # Tokens
        tokens_in = sum(int(r.get("tokens_in", 0)) for r in successful)
        tokens_out = sum(int(r.get("tokens_out", 0)) for r in successful)
        tokens_total = tokens_in + tokens_out

        # Decisions
        models: Dict[str, int] = {}
        devices: Dict[str, int] = {}
        precisions: Dict[str, int] = {}
        tool_counts: Dict[str, int] = {}

        cache_hits = 0
        speculation_count = 0
        reasoning_count = 0
        early_exits = 0

        for r in successful:
            m = r.get("model", "unknown")
            models[m] = models.get(m, 0) + 1

            d = r.get("device", "unknown")
            devices[d] = devices.get(d, 0) + 1

            p = r.get("precision", "unknown")
            precisions[p] = precisions.get(p, 0) + 1

            if r.get("cache_hit"):
                cache_hits += 1
            if r.get("speculation_used"):
                speculation_count += 1
            if r.get("reasoning_used") or r.get("reasoning_strategy"):
                reasoning_count += 1
            if float(r.get("depth", 32)) < 32:
                early_exits += 1

            t = r.get("tool_used")
            if t and t != "none":
                tool_counts[t] = tool_counts.get(t, 0) + 1

        n_succ = len(successful)
        cache_hit_rate = (cache_hits / n_succ) if n_succ > 0 else 0.0
        speculation_rate = (speculation_count / n_succ) if n_succ > 0 else 0.0

        # Percentile calculations
        def percentiles(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
            np_arr = np.array(arr)
            return {
                "mean": round(float(np.mean(np_arr)), 3),
                "median": round(float(np.median(np_arr)), 3),
                "p50": round(float(np.percentile(np_arr, 50)), 3),
                "p90": round(float(np.percentile(np_arr, 90)), 3),
                "p95": round(float(np.percentile(np_arr, 95)), 3),
                "p99": round(float(np.percentile(np_arr, 99)), 3),
                "min": round(float(np.min(np_arr)), 3),
                "max": round(float(np.max(np_arr)), 3),
            }

        total_cost = sum(costs)
        total_energy = sum(energies)
        mean_cost = (total_cost / n_succ) if n_succ > 0 else 0.0
        mean_quality = np.mean([float(r.get("quality_estimate", 1.0)) for r in successful]) if n_succ > 0 else 1.0
        quality_per_dollar = round(float(mean_quality / max(1e-7, mean_cost)), 1) if n_succ > 0 else 0.0

        return {
            "requests": {
                "total": n,
                "successful": n_succ,
                "failed": len(failed),
                "timeouts": len(timeouts),
                "active": 0
            },
            "latency": percentiles(lats),
            "controller_overhead": percentiles(overheads),
            "tokens": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_total
            },
            "economics": {
                "total_cost_usd": round(total_cost, 7),
                "mean_cost_usd": round(mean_cost, 7),
                "total_energy_joules": round(total_energy, 4),
                "quality_per_dollar": quality_per_dollar,
                "estimated_compute_saved_flops": round(sum(float(r.get("compute_saved_flops", 0)) for r in successful), 1)
            },
            "decisions": {
                "models": models,
                "devices": devices,
                "precisions": precisions,
                "mean_depth": round(float(np.mean(depths)), 1) if depths else 32.0,
                "early_exits_count": early_exits,
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "tools_used": tool_counts,
                "speculation_count": speculation_count,
                "speculation_rate": round(speculation_rate, 4),
                "reasoning_count": reasoning_count
            },
            "provenance": {
                "latency": "HYBRID",
                "cost": "ESTIMATED",
                "energy": "ESTIMATED",
                "flops": "DERIVED",
                "decisions": "MEASURED"
            }
        }

    def _empty_summary(self) -> Dict[str, Any]:
        return {
            "requests": {"total": 0, "successful": 0, "failed": 0, "timeouts": 0, "active": 0},
            "latency": {"mean": 0.0, "median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0},
            "controller_overhead": {"mean": 0.0, "median": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0},
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "economics": {"total_cost_usd": 0.0, "mean_cost_usd": 0.0, "total_energy_joules": 0.0, "quality_per_dollar": 0.0, "estimated_compute_saved_flops": 0.0},
            "decisions": {
                "models": {},
                "devices": {},
                "precisions": {},
                "mean_depth": 32.0,
                "early_exits_count": 0,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "tools_used": {},
                "speculation_count": 0,
                "speculation_rate": 0.0,
                "reasoning_count": 0
            },
            "provenance": {
                "latency": "HYBRID",
                "cost": "ESTIMATED",
                "energy": "ESTIMATED",
                "flops": "DERIVED",
                "decisions": "MEASURED"
            }
        }


def format_prometheus_metrics(summary: Dict[str, Any]) -> str:
    """Format summary metrics in Prometheus exposition text format."""
    lines = [
        "# HELP aeon_requests_total Total number of inference requests handled",
        "# TYPE aeon_requests_total counter",
        f"aeon_requests_total {summary['requests']['total']}",
        "# HELP aeon_requests_successful Successful inference requests",
        "# TYPE aeon_requests_successful counter",
        f"aeon_requests_successful {summary['requests']['successful']}",
        "# HELP aeon_requests_failed Failed inference requests",
        "# TYPE aeon_requests_failed counter",
        f"aeon_requests_failed {summary['requests']['failed']}",
        "# HELP aeon_tokens_total Total tokens processed",
        "# TYPE aeon_tokens_total counter",
        f"aeon_tokens_total {summary['tokens']['total_tokens']}",
        "# HELP aeon_latency_ms_p50 Median latency in milliseconds",
        "# TYPE aeon_latency_ms_p50 gauge",
        f"aeon_latency_ms_p50 {summary['latency']['p50']}",
        "# HELP aeon_latency_ms_p95 95th percentile latency in milliseconds",
        "# TYPE aeon_latency_ms_p95 gauge",
        f"aeon_latency_ms_p95 {summary['latency']['p95']}",
        "# HELP aeon_cost_usd_total Estimated total monetary cost in USD",
        "# TYPE aeon_cost_usd_total counter",
        f"aeon_cost_usd_total {summary['economics']['total_cost_usd']}",
        "# HELP aeon_cache_hit_rate Rate of requests served from cache",
        "# TYPE aeon_cache_hit_rate gauge",
        f"aeon_cache_hit_rate {summary['decisions']['cache_hit_rate']}",
        "# HELP aeon_mean_depth Mean layers executed",
        "# TYPE aeon_mean_depth gauge",
        f"aeon_mean_depth {summary['decisions']['mean_depth']}",
    ]
    return "\n".join(lines) + "\n"
