"""
AEON Phase 9 — Result Export Subsystem.

Exports immutable benchmark results to machine-readable JSON and CSV formats
alongside the cryptographic execution manifest.
"""

import csv
import json
import os
from typing import Dict, Any
from aeon.benchmarks.types import BenchmarkRun


def export_benchmark_run(run: BenchmarkRun, output_dir: str = "artifacts/phase9") -> Dict[str, str]:
    """Serialize benchmark manifest, summary results JSON, and trial CSV."""
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "benchmark_manifest.json")
    results_path = os.path.join(output_dir, "results.json")
    csv_path = os.path.join(output_dir, "results.csv")

    # 1. Export Manifest
    manifest_dict = run.manifest.model_dump() if hasattr(run.manifest, "model_dump") else run.manifest.dict()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    # 2. Export Results JSON
    results_map = {}
    for k, v in run.results.items():
        results_map[k] = v.model_dump() if hasattr(v, "model_dump") else v.dict()
    comp_map = {}
    for k, v in run.comparisons.items():
        comp_map[k] = v.model_dump() if hasattr(v, "model_dump") else v.dict()

    results_dict = {
        "run_id": run.run_id,
        "manifest": manifest_dict,
        "results": results_map,
        "comparisons": comp_map
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    # 3. Export Trials CSV
    if run.trials:
        headers = [
            "trial_id", "workload_id", "config_name", "correct", "latency_ms",
            "cost_usd", "tokens", "flops", "memory_mb", "energy_joules",
            "utility", "depth_layers", "device", "precision", "quantization", "overhead_ms"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for t in run.trials:
                writer.writerow([
                    t.trial_id, t.workload_id, t.config_name, t.correct, t.latency_ms,
                    t.cost_usd, t.tokens, t.flops, t.memory_mb, t.energy_joules,
                    t.utility, t.depth_layers, t.device, t.precision, t.quantization, t.controller_overhead_ms
                ])

    return {
        "manifest": manifest_path,
        "results_json": results_path,
        "results_csv": csv_path
    }
