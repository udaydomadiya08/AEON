"""
AEON Phase 9 — Benchmark CLI Subsystem.

Provides full click CLI subcommands for executing, listing, comparing,
and reporting benchmarks.
"""

import click
import json
from typing import Optional
from aeon.benchmarks.types import BenchmarkConfig, ExecutionMode
from aeon.benchmarks.runner import BenchmarkRunner
from aeon.benchmarks.workload_registry import global_workload_registry
from aeon.benchmarks.baselines import get_canonical_baselines
from aeon.benchmarks.ablations import get_canonical_ablations
from aeon.benchmarks.export import export_benchmark_run
from aeon.benchmarks.report import generate_phase9_markdown_report
from aeon.benchmarks.integrity import BenchmarkIntegrityScanner
from aeon.benchmarks.reproducibility import verify_runs_identical
from aeon.benchmarks.sensitivity import SensitivityAnalysisEngine
from aeon.benchmarks.comparisons import compute_pareto_frontier


@click.group(name="benchmark", invoke_without_command=True)
@click.pass_context
def benchmark_group(ctx):
    """AEON Phase 9 Benchmark & Ablation Management."""
    if ctx.invoked_subcommand is None:
        click.echo("AEON Benchmark Suite. Use 'aeon benchmark --help' to view available subcommands.")


@benchmark_group.command("list")
def benchmark_list():
    """List registered canonical workloads, baselines, and ablations."""
    click.echo("\nCanonical Workloads:")
    for w in global_workload_registry.list_workloads():
        click.echo(f"  [{w.id:<14}] {w.name:<32} (Family: {w.family.value}, Diff: {w.declared_difficulty:.2f})")

    click.echo("\nCanonical Baselines:")
    for b in get_canonical_baselines():
        click.echo(f"  [{b.id:<18}] {b.name}")

    click.echo("\nCanonical Ablations:")
    for a in get_canonical_ablations():
        click.echo(f"  [{a.id:<22}] {a.name} (Disables: {a.target_component})")
    click.echo()


@benchmark_group.command("run")
@click.option("--mode", type=click.Choice(["simulated", "measured", "hybrid"]), default="hybrid", help="Execution mode")
@click.option("--seed", default=42, type=int, help="Deterministic random seed")
@click.option("--trials", default=1, type=int, help="Repetitions per workload")
@click.option("--sla", default=8000.0, type=float, help="SLA latency in ms")
@click.option("--budget", default=0.05, type=float, help="Cost budget in USD")
@click.option("--output", default="artifacts/phase9", help="Output directory")
def benchmark_run(mode: str, seed: int, trials: int, sla: float, budget: float, output: str):
    """Run full canonical benchmark suite across all baselines and ablations."""
    click.echo(f"\nStarting AEON Phase 9 Benchmark Suite (Mode: {mode.upper()}, Seed: {seed}, Trials: {trials})...")
    config = BenchmarkConfig(
        mode=ExecutionMode(mode),
        seed=seed,
        repetitions=trials,
        sla_ms=sla,
        cost_budget_usd=budget
    )
    runner = BenchmarkRunner(config)
    run_result = runner.run_suite()

    click.echo("\nExporting results...")
    export_paths = export_benchmark_run(run_result, output_dir=output)
    click.echo(f"  - Manifest: {export_paths['manifest']}")
    click.echo(f"  - Results:  {export_paths['results_json']}")
    click.echo(f"  - CSV:      {export_paths['results_csv']}")

    click.echo("\nBenchmark completed successfully.")


@benchmark_group.command("compare")
@click.option("--baseline", default="B0_direct", help="Baseline condition name")
@click.option("--candidate", default="A0_full_aeon", help="Candidate condition name")
def benchmark_compare(baseline: str, candidate: str):
    """Compare candidate condition against baseline."""
    click.echo(f"\nComparing '{candidate}' against '{baseline}'...")
    runner = BenchmarkRunner()
    run_result = runner.run_suite()
    cand_res = run_result.results.get(candidate)
    base_res = run_result.results.get(baseline)
    if not cand_res or not base_res:
        click.echo("Error: Condition not found in benchmark run.")
        return
    click.echo(f"Quality Delta:     {(cand_res.quality - base_res.quality)*100:+.2f}%")
    click.echo(f"Latency Delta:     {cand_res.mean_latency_ms - base_res.mean_latency_ms:+.2f}ms")
    click.echo(f"Cost Delta:        ${cand_res.mean_cost_usd - base_res.mean_cost_usd:+.7f}")
    click.echo(f"Quality/$ Delta:   {cand_res.quality_per_dollar - base_res.quality_per_dollar:+,.1f}")


@benchmark_group.command("ablation")
def benchmark_ablation():
    """Run canonical 10-way ablation suite and report results."""
    click.echo("\nRunning AEON Canonical 10-Way Ablation Study...")
    runner = BenchmarkRunner()
    run_result = runner.run_suite()
    click.echo(f"{'Condition':<25} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Quality/$':<11}")
    click.echo("-" * 72)
    for k, v in run_result.results.items():
        if k.startswith("A"):
            click.echo(f"{v.condition_name:<25} | {v.quality*100:>6.1f}% | {v.mean_latency_ms:>8.2f}ms | ${v.mean_cost_usd:>9.6f} | {v.quality_per_dollar:>11,f}")
    click.echo()


@benchmark_group.command("sensitivity")
def benchmark_sensitivity():
    """Run multidimensional parameter sweeps over utility weights."""
    click.echo("\nRunning Sensitivity Sweep across latency weight lambda_latency (0.05 to 0.65)...")
    candidates = [
        {"device": "cpu", "precision": "fp32", "quality": 1.0, "latency_ms": 100.0, "cost_usd": 4e-5, "memory_mb": 5000.0, "energy_joules": 6.0},
        {"device": "mps", "precision": "fp16", "quality": 1.0, "latency_ms": 10.0, "cost_usd": 7e-6, "memory_mb": 2500.0, "energy_joules": 0.3},
        {"device": "mps", "precision": "int8", "quality": 0.85, "latency_ms": 4.0, "cost_usd": 3e-6, "memory_mb": 1400.0, "energy_joules": 0.15},
    ]
    from aeon.benchmarks.validation import calculate_independent_utility
    def eval_u(cand, weights):
        return calculate_independent_utility(
            quality=cand["quality"],
            latency_ms=cand["latency_ms"],
            cost_usd=cand["cost_usd"],
            memory_mb=cand["memory_mb"],
            energy_joules=cand["energy_joules"],
            weights=weights
        )
    sweeps = SensitivityAnalysisEngine.sweep_latency_weight(eval_u, candidates)
    click.echo(f"{'lambda_latency':<16} | {'Selected Device':<15} | {'Selected Precision':<18} | {'Utility':<8}")
    click.echo("-" * 65)
    for s in sweeps:
        click.echo(f"{s['weight_latency']:<16.2f} | {s['selected_device']:<15} | {s['selected_precision']:<18} | {s['utility']:<8.4f}")
    click.echo()


@benchmark_group.command("pareto")
def benchmark_pareto():
    """Compute and display multi-objective Pareto frontier across candidates."""
    click.echo("\nComputing Pareto Frontier...")
    runner = BenchmarkRunner()
    run_result = runner.run_suite()
    candidates = list(run_result.results.values())
    pareto, dominated = compute_pareto_frontier(candidates)
    click.echo(f"Pareto Optimal Points ({len(pareto)}):")
    for p in pareto:
        click.echo(f"  * {p.condition_name:<25} (Q={p.quality*100:.1f}%, Lat={p.mean_latency_ms:.1f}ms, Cost=${p.mean_cost_usd:.6f})")
    click.echo(f"Dominated Points ({len(dominated)}):")
    for d in dominated:
        click.echo(f"  - {d.condition_name:<25}")
    click.echo()


@benchmark_group.command("reproducibility")
def benchmark_reproducibility():
    """Verify exact bit-for-bit reproducibility over deterministic seed runs."""
    runner_1 = BenchmarkRunner()
    run_1 = runner_1.run_suite()
    runner_2 = BenchmarkRunner()
    run_2 = runner_2.run_suite()

    dict_1 = {k: v.model_dump() if hasattr(v, "model_dump") else v.dict() for k, v in run_1.results.items()}
    dict_2 = {k: v.model_dump() if hasattr(v, "model_dump") else v.dict() for k, v in run_2.results.items()}

    all_identical = True
    for k in dict_1:
        identical, diffs = verify_runs_identical(dict_1[k], dict_2[k])
        if not identical:
            all_identical = False
            click.echo(f"Discrepancy in {k}: {diffs}")

    if all_identical:
        click.echo("REPRODUCIBILITY CHECK: PASS (Identical bit-for-bit results across runs).")
    else:
        click.echo("REPRODUCIBILITY CHECK: FAIL")


@benchmark_group.command("validate")
def benchmark_validate():
    """Run static integrity scan and AST checks."""
    click.echo("\nRunning AEON Benchmark Integrity Scanner...")
    scanner = BenchmarkIntegrityScanner("aeon")
    res = scanner.run_full_scan()
    if res["passed"]:
        click.echo("INTEGRITY SCAN: PASS (Zero hardcoded constants or shortcuts detected).")
    else:
        click.echo(f"INTEGRITY SCAN: FAIL ({len(res['findings'])} findings detected)")
        for f in res["findings"]:
            click.echo(f"  - {f}")


@benchmark_group.command("report")
@click.option("--output", default="docs/AEON_PHASE9_REPORT.md", help="Report output file path")
def benchmark_report(output: str):
    """Generate formal Phase 9 research report."""
    click.echo(f"\nGenerating formal benchmark report to {output}...")
    runner = BenchmarkRunner()
    run_result = runner.run_suite()
    content = generate_phase9_markdown_report(run_result)
    with open(output, "w", encoding="utf-8") as f:
        f.write(content)
    click.echo(f"Report successfully written to {output}.")


@benchmark_group.command("export")
@click.option("--output", default="artifacts/phase9", help="Output directory")
def benchmark_export(output: str):
    """Export benchmark run to JSON manifest and CSV."""
    click.echo(f"\nRunning benchmark and exporting to {output}...")
    runner = BenchmarkRunner()
    run_result = runner.run_suite()
    paths = export_benchmark_run(run_result, output_dir=output)
    click.echo(f"Exported successfully to {output}.")
