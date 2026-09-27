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
@click.option("--sla", default=2000.0, type=float, help="SLA latency in milliseconds")
@click.option("--cascade", is_flag=True, help="Enable verification-based model cascade")
def run(query: str, context: str, sla: float, cascade: bool):
    """Run inference on a query with AEON adaptive compute optimization."""
    aeon = AEON(use_cascade=cascade)
    result = aeon.run(query=query, context=context, sla_latency_ms=sla)

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
def benchmark():
    """Run full comparative benchmark across baselines and AEON adaptive routing."""
    click.echo("\nRunning AEON Phase 1 Comparative Benchmark Suite...\n")
    harness = BenchmarkHarness()
    results = harness.run_all()

    click.echo("=" * 105)
    click.echo(
        f"{'System / Configuration':<30} | {'Accuracy':<8} | {'Latency':<10} | {'Tokens':<8} | {'Cost ($)':<10} | {'Quality/$':<12} | {'Quality/FLOP':<12}"
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


if __name__ == "__main__":
    cli()
