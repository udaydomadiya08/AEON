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
@click.option("--memory", is_flag=True, default=False, help="Enable AEON multi-level memory hierarchy")
@click.option("--memory-policy", default="balanced", type=click.Choice(["balanced", "latency_first", "cost_first", "quality_first"]), help="Memory hierarchy management policy")
@click.option("--tools", is_flag=True, default=False, help="Enable AEON tool-aware computation (Calculator, Python, SymPy, Retrieval, Structured)")
@click.option("--trace", is_flag=True, help="Display detailed step-by-step controller VOI, memory, and tool trace")
def run(query: str, context: str, sla: float, budget: float, policy: str, cascade: bool, target_quality: float, memory: bool, memory_policy: str, tools: bool, trace: bool):
    """Run inference on a query with AEON adaptive compute optimization."""
    chosen_policy = "cascade" if cascade else policy
    from aeon.memory.hierarchy import global_memory_hierarchy
    from aeon.tools.tool_controller import global_tool_controller
    aeon = AEON(policy=chosen_policy, memory=memory, memory_policy=memory_policy, memory_hierarchy=global_memory_hierarchy, tools=tools, tool_controller=global_tool_controller)
    result = aeon.run(
        query=query,
        context=context,
        sla_latency_ms=sla,
        cost_budget_usd=budget or 0.01,
        target_quality=target_quality,
        memory=memory,
        tools=tools,
    )

    if trace:
        click.echo("\n==================== AEON TRACE ====================")
        click.echo(f"Task:              {result.trace.task_type}")
        if result.plan:
            click.echo(f"Difficulty:        {result.plan.difficulty:.2f} ({result.plan.difficulty_level.value})")

        if memory or result.trace.memory_hit:
            click.echo("\n────────────── AEON MEMORY TRACE ──────────────")
            click.echo(f"Memory Hit:        {result.trace.memory_hit}")
            click.echo(f"Memory Level:      {result.trace.memory_level or 'MISS'}")
            click.echo(f"Memory Action:     {result.trace.memory_action or 'COMPUTE'}")
            click.echo(f"Reused Tokens:     {result.trace.reused_tokens}")
            click.echo(f"Estimated Saved:   {result.trace.compute_saved_flops:.2e} FLOPs | ${result.trace.cost_saved_usd:.6f} | {result.trace.latency_saved_ms:.1f}ms")
            click.echo(f"Lookup Latency:    {result.trace.lookup_latency_ms:.2f} ms")
            if result.trace.reuse_decision:
                click.echo(f"Reuse Decision:    {result.trace.reuse_decision}")
            click.echo("────────────────────────────────────────────────")

        if tools or result.trace.tool_used:
            click.echo("\n──────────── AEON TOOL COMPUTATION ────────────")
            click.echo(f"Tool Used:         {result.trace.tool_used or 'NONE'} ({result.trace.tool_category or 'none'})")
            click.echo(f"Verification:      {result.trace.verification_status or 'none'}")
            click.echo(f"Tool Latency:      {result.trace.tool_latency_ms:.2f} ms")
            click.echo(f"Tool Cost:         ${result.trace.tool_cost_usd:.7f} USD")
            click.echo(f"Tool FLOPs:        {result.trace.tool_compute_flops:.2e}")
            click.echo(f"Model Avoided:     {result.trace.unnecessary_model_avoided}")
            if result.trace.plan_steps:
                click.echo("Plan Steps:")
                for s in result.trace.plan_steps:
                    click.echo(f"  - Step {s.get('step_id')}: {s.get('tool')} ({s.get('action')}) -> {s.get('status')}")
            click.echo("────────────────────────────────────────────────")

        if result.trace.trace_steps:
            click.echo("\n────────────── COMPUTE CONTROLLER ──────────────")
            for step in result.trace.trace_steps:
                click.echo(f"Iteration:         {step.get('iteration')}")
                click.echo(f"Model:             {step.get('model')} ({step.get('reasoning')})")
                click.echo(f"Confidence:        {step.get('confidence'):.2f}")
                click.echo(f"Estimated Quality: {step.get('quality_estimate'):.2f}")
                click.echo(f"Marginal Gain:     +{step.get('marginal_gain'):.4f}")
                click.echo(f"Marginal Cost:     ${step.get('marginal_cost'):.6f}")
                click.echo(f"VOI Ratio:         {step.get('voi'):.2f}")
                click.echo(f"Decision:          {step.get('decision')} ({step.get('decision_reason')})")
                click.echo("────────────────────────────────────────────────")

        click.echo(f"FINAL:")
        click.echo(f"Model:             {result.trace.model}")
        click.echo(f"Final Quality:     {result.trace.final_quality_estimate or result.confidence:.2f}")
        click.echo(f"Total Iterations:  {result.trace.number_of_controller_iterations}")
        click.echo(f"Stopping Reason:   {result.trace.stopping_reason or result.trace.decision_reason}")
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
    click.echo(f"Cache Hit:  {result.trace.cache_hit} ({result.trace.memory_level or 'NONE'})")
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


@cli.group()
def memory():
    """AEON Multi-Tier Memory Hierarchy inspection and management (Section 33)."""
    pass


@memory.command("stats")
def memory_stats():
    """Display memory hierarchy utilization and efficiency metrics."""
    from aeon.memory.hierarchy import global_memory_hierarchy
    stats = global_memory_hierarchy.get_stats()
    click.echo("\n==================== AEON MEMORY STATS ====================")
    click.echo(f"Total Lookups:        {stats.total_lookups}")
    click.echo(f"Total Cache Hits:     {stats.total_hits} (Hit Rate: {stats.hit_rate*100:.1f}%)")
    click.echo(f"L1 KV Hits / Misses:  {stats.l1_hits} / {stats.l1_misses}")
    click.echo(f"L2 Prefix Hits:       {stats.l2_hits} / {stats.l2_misses}")
    click.echo(f"L3 Intermediate Hits: {stats.l3_hits} / {stats.l3_misses}")
    click.echo(f"L4 Semantic Hits:     {stats.l4_hits} / {stats.l4_misses}")
    click.echo(f"Total Bytes Used:     {stats.total_bytes_used} bytes")
    click.echo(f"Compute Saved:        {stats.total_compute_saved_flops:.2e} FLOPs")
    click.echo(f"Cost Saved:           ${stats.total_cost_saved_usd:.6f} USD")
    click.echo(f"Latency Saved:        {stats.total_latency_saved_ms:.1f} ms")
    click.echo(f"Tokens Saved:         {stats.total_tokens_saved}")
    click.echo(f"Evictions Count:      {stats.evictions_count}")
    click.echo(f"Admissions (Acc/Rej): {stats.admissions_count} / {stats.admissions_rejected_count}")
    click.echo(f"False Hits Prevented: {stats.false_positives_prevented}")
    click.echo(f"Stale Hits Rejected:  {stats.stale_hits_rejected}")
    click.echo("===========================================================\n")


@memory.command("clear")
def memory_clear():
    """Clear all memory hierarchy tiers (L1, L2, L3, L4)."""
    from aeon.memory.hierarchy import global_memory_hierarchy
    global_memory_hierarchy.clear()
    click.echo("AEON memory hierarchy cleared successfully across all tiers (L1, L2, L3, L4).")


@memory.command("inspect")
def memory_inspect():
    """Detailed inspection of memory contents across tiers."""
    from aeon.memory.hierarchy import global_memory_hierarchy
    inspection = global_memory_hierarchy.inspect()
    click.echo(json.dumps(inspection, indent=2))


@memory.command("benchmark")
def memory_benchmark():
    """Run full Phase 3 memory hierarchy benchmark suite."""
    harness = BenchmarkHarness()
    click.echo("\nRunning AEON Phase 3 Memory Hierarchy Benchmark Suite...\n")
    results = harness.run_phase3()
    click.echo("=" * 115)
    click.echo(
        f"{'Configuration':<26} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Hit Rate':<9} | {'Reuse %':<8} | {'Quality/$':<11} | {'Quality/TFLOP':<13}"
    )
    click.echo("=" * 115)
    for name, d in results.items():
        lat_str = f"{d['mean_latency_ms']:.1f}ms"
        cost_str = f"${d['total_cost_usd']:.5f}"
        click.echo(
            f"{name:<26} | {d['quality']*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {d['hit_rate']*100:>8.1f}% | {d['reuse_ratio']*100:>7.1f}% | {d['quality_per_dollar']:>11.0f} | {d['quality_per_tflops']:>13.4f}"
        )
    click.echo("=" * 115)
    click.echo()


@cli.command()
@click.option("--phase", type=click.Choice(["1", "2", "3", "4"]), default="2", help="Benchmark phase (1, 2, 3, or 4)")
@click.option("--ablation", is_flag=True, help="Run Phase 2 VOI systematic ablation study")
@click.option("--memory-ablation", is_flag=True, help="Run Phase 3 memory tier ablation study")
@click.option("--tool-ablation", is_flag=True, help="Run Phase 4 tool ablation study")
@click.option("--tool-vs-model", is_flag=True, help="Run Section 31 Model vs Tool experiment")
@click.option("--tool-failure", is_flag=True, help="Run Section 33 Tool failure and fallback recovery experiment")
@click.option("--budget-study", is_flag=True, help="Run Section 34 budget sensitivity experiment")
@click.option("--target-exp", is_flag=True, help="Run target quality (>=0.85) experiment")
@click.option("--semantic-safety", is_flag=True, help="Run Section 27 semantic cache false-positive safety experiment")
@click.option("--freshness", is_flag=True, help="Run Section 28 freshness experiment")
@click.option("--memory-pressure", is_flag=True, help="Run Section 29 memory pressure experiment")
@click.option("--voi", is_flag=True, help="Run Phase 2 VOI evaluation")
def benchmark(phase: str, ablation: bool, memory_ablation: bool, tool_ablation: bool, tool_vs_model: bool, tool_failure: bool, budget_study: bool, target_exp: bool, semantic_safety: bool, freshness: bool, memory_pressure: bool, voi: bool):
    """Run comparative benchmark suite and ablations."""
    harness = BenchmarkHarness()

    if tool_ablation:
        click.echo("\nRunning AEON Phase 4 Tool Ablation Study...\n")
        results = harness.run_tool_ablation_study()
        click.echo("=" * 95)
        click.echo(
            f"{'Ablation Configuration':<28} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Tool Calls':<10} | {'Quality/$':<11}"
        )
        click.echo("=" * 95)
        for name, d in results.items():
            lat_str = f"{d['latency_ms']:.1f}ms"
            cost_str = f"${d['cost_usd']:.5f}"
            click.echo(
                f"{name:<28} | {d['quality']*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {d['tool_calls']:>10} | {d['quality_per_dollar']:>11.0f}"
            )
        click.echo("=" * 95)
        click.echo()
        return

    if tool_vs_model:
        click.echo("\nRunning Section 31 Model vs Tool Experiment (Unnecessary Model Invocation Rate)...\n")
        res = harness.run_tool_vs_model_experiment()
        click.echo(json.dumps(res, indent=2))
        return

    if tool_failure:
        click.echo("\nRunning Section 33 Tool Failure and Graceful Fallback Experiment...\n")
        res = harness.run_tool_failure_experiment()
        click.echo(json.dumps(res, indent=2))
        return

    if budget_study:
        click.echo("\nRunning Section 34 Budget Sensitivity Experiment...\n")
        res = harness.run_budget_experiment()
        click.echo(json.dumps(res, indent=2))
        return

    if memory_ablation:
        click.echo("\nRunning AEON Phase 3 Memory Hierarchy Ablation Study...\n")
        results = harness.run_memory_ablation_study()
        click.echo("=" * 105)
        click.echo(
            f"{'Ablation Configuration':<36} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Hit Rate':<9} | {'Quality/$':<11}"
        )
        click.echo("=" * 105)
        for name, d in results.items():
            lat_str = f"{d['mean_latency_ms']:.1f}ms"
            cost_str = f"${d['total_cost_usd']:.5f}"
            click.echo(
                f"{name:<36} | {d['quality']*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {d['hit_rate']*100:>8.1f}% | {d['quality_per_dollar']:>11.0f}"
            )
        click.echo("=" * 105)
        click.echo()
        return

    if semantic_safety:
        click.echo("\nRunning Section 27 Semantic Cache False-Positive Safety Experiment...\n")
        res = harness.run_semantic_safety_experiment()
        click.echo(json.dumps(res, indent=2))
        return

    if freshness:
        click.echo("\nRunning Section 28 Freshness Decay Experiment...\n")
        res = harness.run_freshness_experiment()
        click.echo(json.dumps(res, indent=2))
        return

    if memory_pressure:
        click.echo("\nRunning Section 29 Memory Pressure Experiment (LRU vs LFU vs Value-Aware)...\n")
        res = harness.run_memory_pressure_experiment()
        click.echo(json.dumps(res, indent=2))
        return

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

    if phase == "4":
        click.echo("\nRunning AEON Phase 4 Tool-Aware Computation Benchmark Suite...\n")
        results = harness.run_phase4()
        click.echo("=" * 115)
        click.echo(
            f"{'Configuration':<32} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Tokens':<8} | {'Quality/$':<11} | {'Quality/TFLOP':<13}"
        )
        click.echo("=" * 115)
        for name, m in results.items():
            lat_str = f"{m.mean_latency_ms:.1f}ms"
            cost_str = f"${m.total_cost_usd:.5f}"
            click.echo(
                f"{name:<32} | {m.accuracy*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {m.total_tokens:>8} | {m.quality_per_dollar:>11.0f} | {m.quality_per_tflops:>13.4f}"
            )
        click.echo("=" * 115)
        click.echo()
        return

    if phase == "3":
        click.echo("\nRunning AEON Phase 3 Memory Hierarchy Benchmark Suite...\n")
        results = harness.run_phase3()
        click.echo("=" * 115)
        click.echo(
            f"{'Configuration':<26} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Hit Rate':<9} | {'Reuse %':<8} | {'Quality/$':<11} | {'Quality/TFLOP':<13}"
        )
        click.echo("=" * 115)
        for name, d in results.items():
            lat_str = f"{d['mean_latency_ms']:.1f}ms"
            cost_str = f"${d['total_cost_usd']:.5f}"
            click.echo(
                f"{name:<26} | {d['quality']*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {d['hit_rate']*100:>8.1f}% | {d['reuse_ratio']*100:>7.1f}% | {d['quality_per_dollar']:>11.0f} | {d['quality_per_tflops']:>13.4f}"
            )
        click.echo("=" * 115)
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


# =========================================================================
# Phase 4 Tool Commands (Section 38)
# =========================================================================

@cli.group()
def tools():
    """AEON Tool-Aware Computation commands."""
    pass


@tools.command(name="list")
def tools_list():
    """List registered computational tools and their capability descriptors."""
    from aeon.tools.registry import global_tool_registry
    tool_names = global_tool_registry.list_tools()
    click.echo("\n==================== AEON REGISTERED TOOLS ====================")
    click.echo(f"{'Tool Name':<15} | {'Category':<16} | {'Deterministic':<14} | {'Mode':<22}")
    click.echo("-" * 72)
    for name in tool_names:
        t = global_tool_registry.get(name)
        mode = getattr(t, "execution_mode", "REAL_EXECUTION")
        det = "True" if getattr(t, "category", None) and t.category.value in ["deterministic", "symbolic"] else "False"
        cat = t.category.value if hasattr(t, "category") else "deterministic"
        click.echo(f"{name:<15} | {cat:<16} | {det:<14} | {mode:<22}")
    click.echo("===============================================================\n")


@tools.command(name="stats")
def tools_stats():
    """Display cumulative tool execution and compute-avoidance telemetry."""
    from aeon.tools.tool_controller import global_tool_controller
    s = global_tool_controller.stats
    click.echo("\n==================== AEON TOOL STATS ====================")
    click.echo(f"Total Tool Calls:          {s.total_calls}")
    click.echo(f"Successful Calls:          {s.successful_calls}")
    click.echo(f"Failed Calls:              {s.failed_calls}")
    click.echo(f"Unnecessary Models Avoided:{s.unnecessary_models_avoided}")
    click.echo(f"Total Compute Saved:       {s.total_compute_saved_flops:.2e} FLOPs")
    click.echo(f"Total Cost Saved:          ${s.total_cost_saved_usd:.6f} USD")
    click.echo(f"Total Latency Saved:       {s.total_latency_saved_ms:.1f} ms")
    click.echo(f"Verification Runs:         {s.verification_runs}")
    click.echo(f"Fallbacks Triggered:       {s.fallbacks_triggered}")
    if s.calls_by_tool:
        click.echo("Calls By Tool:")
        for t_name, c in s.calls_by_tool.items():
            click.echo(f"  - {t_name:<16}: {c}")
    click.echo("=========================================================\n")


@tools.command(name="clear")
def tools_clear():
    """Reset operational tool execution statistics."""
    from aeon.tools.tool_controller import global_tool_controller
    global_tool_controller.clear_stats()
    click.echo("Tool operational telemetry reset successfully.")


@tools.command(name="benchmark")
def tools_benchmark():
    """Run dedicated Phase 4 tool experiments (model vs tool and fallback tests)."""
    harness = BenchmarkHarness()
    click.echo("\n--- Running Section 31 Model vs Tool Experiment ---")
    tvm = harness.run_tool_vs_model_experiment()
    click.echo(f"Unnecessary Model Invocation Rate: {tvm['unnecessary_model_invocation_rate']}")
    click.echo(f"Tool Routing Accuracy:             {tvm['tool_routing_accuracy']*100:.1f}%")
    click.echo(f"Status:                            {tvm['status']}")

    click.echo("\n--- Running Section 33 Tool Failure Recovery Experiment ---")
    tfe = harness.run_tool_failure_experiment()
    click.echo(f"Fallback Succeeded:                {tfe['fallback_succeeded']}")
    click.echo(f"Fallback Route:                    {tfe['fallback_route']}")
    click.echo(f"Status:                            {tfe['status']}\n")


if __name__ == "__main__":
    cli()
