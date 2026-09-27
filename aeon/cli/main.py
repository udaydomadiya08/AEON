"""
AEON Command Line Interface.
Supports:
  aeon run "<query>"
  aeon benchmark
  aeon models
  aeon profile
"""

import json
import click
from aeon import AEON
from aeon.models.model_registry import global_registry
from aeon.benchmarks.harness import BenchmarkHarness


@click.group()
def cli():
    """AEON — Adaptive Economically-Optimized Network CLI."""
    pass


@cli.command()
@click.argument("query")
@click.option("--context", default=None, help="Optional context for query")
@click.option("--sla", "--max-latency", "sla", default=8000.0, type=float, help="SLA latency in milliseconds")
@click.option("--budget", default=None, type=float, help="Maximum cost budget in USD")
@click.option("--policy", type=click.Choice(["voi", "rules", "cascade"]), default="voi", help="Controller policy")
@click.option("--cascade", is_flag=True, help="Enable verification-based model cascade")
@click.option("--target-quality", default=None, type=float, help="Target quality score (0.0 to 1.0)")
@click.option("--trace", is_flag=True, help="Display detailed step-by-step controller VOI trace")
def run(query: str, context: str, sla: float, budget: float, policy: str, cascade: bool, target_quality: float, trace: bool):
    """Run inference on a query with AEON adaptive compute optimization."""
    chosen_policy = "cascade" if cascade else policy
    aeon = AEON(policy=chosen_policy)
    result = aeon.run(
        query=query,
        context=context,
        sla_latency_ms=sla,
        cost_budget_usd=budget or 0.01,
        target_quality=target_quality,
    )

    if trace and result.trace.trace_steps:
        click.echo("\n==================== AEON TRACE ====================")
        click.echo(f"Task:              {result.trace.task_type}")
        click.echo(f"Difficulty:        {result.plan.difficulty:.2f} ({result.plan.difficulty_level.value})")
        click.echo("────────────────────────────────────────────────────")
        for step in result.trace.trace_steps:
            click.echo(f"Iteration:         {step.get('iteration')}")
            click.echo(f"Model:             {step.get('model')} ({step.get('reasoning')})")
            click.echo(f"Confidence:        {step.get('confidence'):.2f}")
            click.echo(f"Estimated Quality: {step.get('quality_estimate'):.2f}")
            click.echo(f"Marginal Gain:     +{step.get('marginal_gain'):.4f}")
            click.echo(f"Marginal Cost:     ${step.get('marginal_cost'):.6f}")
            click.echo(f"VOI Ratio:         {step.get('voi'):.2f}")
            click.echo(f"Decision:          {step.get('decision')} ({step.get('decision_reason')})")
            click.echo("────────────────────────────────────────────────────")
        click.echo(f"FINAL:")
        click.echo(f"Model:             {result.trace.model}")
        click.echo(f"Final Quality:     {result.trace.final_quality_estimate or result.confidence:.2f}")
        click.echo(f"Total Iterations:  {result.trace.number_of_controller_iterations}")
        click.echo(f"Stopping Reason:   {result.trace.stopping_reason}")
        click.echo(f"Total Compute:     {result.compute:.2e} FLOPs")
        click.echo(f"Total Cost:        ${result.cost:.7f} USD")
        click.echo(f"Total Latency:     {result.latency_ms:.2f} ms")
        click.echo("====================================================\n")

    click.echo("\n==================== AEON RESULT ====================")
    click.echo(f"Answer:     {result.answer}")
    click.echo(f"Route:      {result.route}")
    click.echo(f"Model:      {result.trace.model}")
    click.echo(f"Reasoning:  {result.trace.reasoning_mode}")
    click.echo(f"Confidence: {result.confidence:.2f}")
    click.echo(f"Latency:    {result.latency_ms:.2f} ms")
    click.echo(f"FLOPs:      {result.compute:.2e}")
    click.echo(f"Cost:       ${result.cost:.7f} USD")
    click.echo("================== EXECUTION PLAN ===================")
    if result.plan:
        click.echo(f"Task Type:  {result.plan.task_type.value}")
        click.echo(f"Difficulty: {result.plan.difficulty:.2f} ({result.plan.difficulty_level.value})")
        click.echo(f"Uncertainty:{result.plan.uncertainty:.2f}")
        click.echo(f"Stop Policy:{result.plan.stop_policy}")
        if result.trace.stopping_reason:
            click.echo(f"Stop Reason:{result.trace.stopping_reason}")
    click.echo("=====================================================\n")


@cli.command()
def models():
    """List registered model profiles and operational characteristics."""
    click.echo("\nRegistered AEON Model Profiles:")
    click.echo(f"{'Name':<20} {'Params (B)':<12} {'Precision':<10} {'$/1M In':<10} {'$/1M Out':<10} {'Base Latency':<12}")
    click.echo("-" * 76)
    for m in global_registry.list_models():
        click.echo(
            f"{m.name:<20} {m.param_count_b:<12.2f} {m.precision:<10} "
            f"${m.cost_per_1m_input_usd:<9.2f} ${m.cost_per_1m_output_usd:<9.2f} {m.base_latency_ms:<12.1f}ms"
        )
    click.echo()


@cli.command()
def profile():
    """Display detected hardware environment and execution backends."""
    import platform
    import sys
    try:
        import torch
        mps = torch.backends.mps.is_available()
        cuda = torch.cuda.is_available()
    except ImportError:
        mps = False
        cuda = False

    click.echo("\nAEON Hardware & Runtime Profile:")
    click.echo(f"Platform:      {platform.platform()} ({platform.machine()})")
    click.echo(f"Python:        {sys.version.split()[0]}")
    click.echo(f"Apple Silicon: {'Yes (MPS available)' if mps else 'No'}")
    click.echo(f"NVIDIA CUDA:   {'Yes' if cuda else 'No'}")
    click.echo("Default Mode:  Development / Resource-Constrained (Adaptive)\n")


@cli.command()
@click.option("--phase", type=click.Choice(["1", "2"]), default="2", help="Benchmark phase (1 or 2)")
@click.option("--ablation", is_flag=True, help="Run systematic ablation study")
@click.option("--target-exp", is_flag=True, help="Run target quality (>=0.85) experiment")
@click.option("--voi", is_flag=True, help="Run Phase 2 VOI evaluation")
def benchmark(phase: str, ablation: bool, target_exp: bool, voi: bool):
    """Run comparative benchmark suite and ablations."""
    harness = BenchmarkHarness()

    if ablation:
        click.echo("\nRunning AEON Phase 2 Systematic Ablation Study...\n")
        results = harness.run_ablation_study()
        click.echo("=" * 105)
        click.echo(
            f"{'Ablation Configuration':<30} | {'Accuracy':<8} | {'Latency':<10} | {'Tokens':<8} | {'Cost ($)':<10} | {'Quality/$':<12} | {'Quality/FLOP':<12}"
        )
        click.echo("=" * 105)
        for name, m in results.items():
            lat_str = f"{m.mean_latency_ms:.1f}ms"
            cost_str = f"${m.total_cost_usd:.5f}"
            click.echo(
                f"{name:<30} | {m.accuracy*100:>7.1f}% | {lat_str:>10} | {m.total_tokens:>8} | {cost_str:>10} | {m.quality_per_dollar:>12.0f} | {m.quality_per_tflops:>12.4f}"
            )
        click.echo("=" * 105)
        click.echo()
        return

    if target_exp:
        click.echo("\nRunning Target Quality Experiment (Target: Quality >= 0.85)...\n")
        exp_results = harness.run_target_quality_experiment(target_quality=0.85)
        click.echo("=" * 95)
        click.echo(
            f"{'System / Configuration':<30} | {'Quality':<8} | {'Met?':<6} | {'Tokens':<8} | {'Cost ($)':<10} | {'Latency':<10} | {'Quality/$':<10}"
        )
        click.echo("=" * 95)
        for name, d in exp_results.items():
            met_str = "YES" if d["target_met"] else "NO"
            lat_str = f"{d['mean_latency_ms']:.1f}ms"
            cost_str = f"${d['total_cost_usd']:.5f}"
            click.echo(
                f"{name:<30} | {d['achieved_quality']*100:>7.1f}% | {met_str:>6} | {d['total_tokens']:>8} | {cost_str:>10} | {lat_str:>10} | {d['quality_per_dollar']:>10.0f}"
            )
        click.echo("=" * 95)
        click.echo()
        return

    if phase == "2" or voi:
        click.echo("\nRunning AEON Phase 2 Comparative Benchmark Suite (with VOI Controller)...\n")
        results = harness.run_phase2()
    else:
        click.echo("\nRunning AEON Phase 1 Comparative Benchmark Suite...\n")
        results = harness.run_all()

    click.echo("=" * 115)
    click.echo(
        f"{'System / Configuration':<30} | {'Accuracy':<8} | {'Latency':<10} | {'Tokens':<8} | {'Cost ($)':<10} | {'Quality/$':<12} | {'Quality/FLOP':<12} | {'Iters':<6}"
    )
    click.echo("=" * 115)
    for name, m in results.items():
        lat_str = f"{m.mean_latency_ms:.1f}ms"
        cost_str = f"${m.total_cost_usd:.5f}"
        click.echo(
            f"{name:<30} | {m.accuracy*100:>7.1f}% | {lat_str:>10} | {m.total_tokens:>8} | {cost_str:>10} | {m.quality_per_dollar:>12.0f} | {m.quality_per_tflops:>12.4f} | {m.mean_iterations:>6.1f}"
        )
    click.echo("=" * 115)
    click.echo()


if __name__ == "__main__":
    cli()
