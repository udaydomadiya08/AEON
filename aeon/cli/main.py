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
@click.option("--speculate", is_flag=True, default=False, help="Enable AEON speculative and token-efficient inference (Phase 5)")
@click.option("--spec-strategy", default=None, type=click.Choice(["standard", "draft_model", "ngram", "multi_token"]), help="Force specific speculation strategy")
@click.option("--draft-model", default=None, type=str, help="Specific draft model name to deploy")
@click.option("--reasoning", is_flag=True, default=False, help="Enable AEON adaptive latent & token-efficient reasoning (Phase 6)")
@click.option("--reason-strategy", default=None, type=click.Choice(["none", "direct", "shallow", "medium", "deep", "latent", "explicit", "tool_assisted", "model_escalation", "hybrid"]), help="Force specific reasoning strategy")
@click.option("--adaptive-depth", "--depth", "adaptive_depth", is_flag=True, default=False, help="Enable AEON adaptive model depth & Mixture-of-Depths (Phase 7)")
@click.option("--depth-strategy", default=None, type=click.Choice(["hybrid_depth", "full_depth", "fixed_early_exit", "adaptive_early_exit", "fixed_mod", "adaptive_mod"]), help="Force specific depth strategy")
@click.option("--max-depth", default=None, type=int, help="Maximum allowed model layer depth")
@click.option("--min-depth", default=None, type=int, help="Minimum allowed model layer depth")
@click.option("--enable-early-exit", is_flag=True, default=False, help="Explicitly enable early exit evaluator")
@click.option("--enable-mod", is_flag=True, default=False, help="Explicitly enable Mixture-of-Depths router")
@click.option("--depth-budget", default=None, type=float, help="Explicit depth layer budget")
@click.option("--hardware-aware", is_flag=True, default=False, help="Enable AEON hardware-aware precision & quantization (Phase 8)")
@click.option("--device", default=None, type=str, help="Preferred execution device (auto, cpu, mps, cuda)")
@click.option("--precision", default=None, type=str, help="Preferred precision (auto, fp32, fp16, bf16, int8, fp8)")
@click.option("--quantization", default=None, type=str, help="Preferred quantization method (auto, none, int8_weight_only, simulated_int8)")
@click.option("--memory-budget", default=None, type=float, help="Explicit memory budget in MB")
@click.option("--trace", is_flag=True, help="Display detailed step-by-step controller VOI, memory, tool, and reasoning trace")
def run(query: str, context: str, sla: float, budget: float, policy: str, cascade: bool, target_quality: float, memory: bool, memory_policy: str, tools: bool, speculate: bool, spec_strategy: str, draft_model: str, reasoning: bool, reason_strategy: str, adaptive_depth: bool, depth_strategy: str, max_depth: int, min_depth: int, enable_early_exit: bool, enable_mod: bool, depth_budget: float, hardware_aware: bool, device: str, precision: str, quantization: str, memory_budget: float, trace: bool):
    """Run inference on a query with AEON adaptive compute optimization."""
    chosen_policy = "cascade" if cascade else policy
    from aeon.memory.hierarchy import global_memory_hierarchy
    from aeon.tools.tool_controller import global_tool_controller
    from aeon.speculation.controller import global_speculation_controller
    from aeon.reasoning.controller import global_reasoning_controller
    from aeon.depth.controller import global_depth_controller
    from aeon.hardware.controller import global_hardware_controller
    aeon = AEON(
        policy=chosen_policy,
        memory=memory,
        memory_policy=memory_policy,
        memory_hierarchy=global_memory_hierarchy,
        tools=tools,
        tool_controller=global_tool_controller,
        speculation=speculate,
        speculation_controller=global_speculation_controller,
        reasoning=reasoning,
        reasoning_controller=global_reasoning_controller,
        adaptive_depth=adaptive_depth,
        depth_controller=global_depth_controller,
        hardware_aware=hardware_aware,
        hardware_controller=global_hardware_controller,
    )
    result = aeon.run(
        query=query,
        context=context,
        sla_latency_ms=sla,
        cost_budget_usd=budget or 0.01,
        target_quality=target_quality,
        memory=memory,
        tools=tools,
        speculation=speculate,
        speculation_strategy=spec_strategy,
        draft_model=draft_model,
        reasoning=reasoning,
        reasoning_strategy=reason_strategy,
        adaptive_depth=adaptive_depth,
        depth_strategy=depth_strategy,
        max_depth=max_depth,
        min_depth=min_depth,
        hardware_aware=hardware_aware,
        device=device,
        precision=precision,
        quantization=quantization,
        memory_budget_mb=memory_budget,
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

        if speculate or result.trace.speculative:
            click.echo("\n──────────── AEON SPECULATIVE INFERENCE ────────────")
            click.echo(f"Strategy:          {result.trace.speculation_strategy or 'standard'}")
            click.echo(f"Action:            {result.trace.speculation_action or 'NONE'}")
            click.echo(f"Draft Model:       {result.trace.draft_model or 'none'} (Length: {result.trace.draft_length})")
            click.echo(f"Drafted Tokens:    {result.trace.drafted_tokens} | Accepted: {result.trace.accepted_tokens} | Rejected: {result.trace.rejected_tokens}")
            click.echo(f"Acceptance Rate:   {result.trace.acceptance_rate*100:.1f}% (Avg/round: {result.trace.average_accepted_per_round})")
            click.echo(f"Draft Latency:     {result.trace.draft_latency_ms:.2f} ms | Verify: {result.trace.verification_latency_ms:.2f} ms")
            click.echo(f"Realized Speedup:  {result.trace.realized_speedup:.2f}x (Expected: {result.trace.expected_speedup:.2f}x)")
            click.echo(f"Break-Even Rate:   {result.trace.break_even_acceptance*100:.1f}%")
            click.echo(f"Execution Mode:    {result.trace.speculation_mode}")
            click.echo("────────────────────────────────────────────────────")

        if reasoning or result.trace.reasoning_enabled:
            click.echo("\n──────────── AEON ADAPTIVE REASONING ────────────")
            click.echo(f"Strategy:          {result.trace.reasoning_strategy or 'none'}")
            click.echo(f"Model:             {result.trace.reasoning_model or result.trace.model}")
            click.echo(f"Steps:             {result.trace.reasoning_depth_steps} (Latent: {result.trace.reasoning_latent_steps}, Explicit: {result.trace.reasoning_explicit_tokens} tokens)")
            click.echo(f"Quality Est:       {result.trace.reasoning_quality_estimate:.2f} | Uncertainty: {result.trace.reasoning_uncertainty:.2f}")
            click.echo(f"Marginal Gain:     +{result.trace.reasoning_marginal_gain:.4f} | Marginal Cost: ${result.trace.reasoning_marginal_cost_usd:.6f}")
            click.echo(f"VOI Ratio:         {result.trace.reasoning_voi:.2f}")
            click.echo(f"Stop Reason:       {result.trace.reasoning_stop_reason or 'NONE'}")
            click.echo(f"Budget Tokens:     {result.trace.reasoning_budget_tokens_used}/{result.trace.reasoning_budget_tokens_max}")
            click.echo(f"Answer Stable:     {result.trace.reasoning_answer_stable}")
            click.echo(f"Execution Mode:    {result.trace.reasoning_execution_mode}")
            click.echo("─────────────────────────────────────────────────")

        if adaptive_depth or getattr(result.trace, "depth_enabled", False):
            click.echo("\n──────────── AEON ADAPTIVE DEPTH & MOD ────────────")
            click.echo(f"Strategy:          {result.trace.depth_strategy or 'hybrid_depth'}")
            click.echo(f"Model Layers:      {result.trace.actual_depth}/{result.trace.requested_depth} (Saved: {result.trace.depth_layers_saved})")
            click.echo(f"Early Exit:        {result.trace.early_exit} (Reason: {result.trace.early_exit_reason or 'NONE'})")
            click.echo(f"MoD Enabled:       {result.trace.mod_enabled} (Capacity: {result.trace.mod_capacity:.2f})")
            click.echo(f"Token Breakdown:   Active: {result.trace.mod_active_tokens} | Deep: {result.trace.mod_deep_tokens} | Shallow: {result.trace.mod_shallow_tokens} | Skipped: {result.trace.mod_skipped_tokens}")
            click.echo(f"Marginal Gain:     +{result.trace.depth_marginal_gain:.4f} | Marginal Cost: ${result.trace.depth_marginal_cost_usd:.6f}")
            click.echo(f"VOI Ratio:         {result.trace.depth_voi:.2f}")
            click.echo(f"Estimated FLOPs:   {result.trace.depth_estimated_flops:.2e}")
            click.echo(f"Controller OH:     {result.trace.depth_controller_overhead_ms:.2f} ms")
            click.echo(f"Execution Mode:    {result.trace.depth_execution_mode}")
            click.echo("────────────────────────────────────────────────────")

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


@cli.group(invoke_without_command=True)
@click.pass_context
@click.option("--phase", type=click.Choice(["1", "2", "3", "4", "5"]), default="2", help="Benchmark phase (1, 2, 3, 4, or 5)")
@click.option("--ablation", is_flag=True, help="Run Phase 2 VOI systematic ablation study")
@click.option("--memory-ablation", is_flag=True, help="Run Phase 3 memory tier ablation study")
@click.option("--tool-ablation", is_flag=True, help="Run Phase 4 tool ablation study")
@click.option("--tool-vs-model", is_flag=True, help="Run Section 31 Model vs Tool experiment")
@click.option("--tool-failure", is_flag=True, help="Run Section 33 Tool failure and fallback recovery experiment")
@click.option("--budget-study", is_flag=True, help="Run Section 34 budget sensitivity experiment")
@click.option("--speculation-ablation", is_flag=True, help="Run Phase 5 speculation component ablation study")
@click.option("--speculation-break-even", is_flag=True, help="Run Phase 5 economic break-even experiment")
@click.option("--speculation-workload", is_flag=True, help="Run Phase 5 workload breakdown across 8 categories")
@click.option("--target-exp", is_flag=True, help="Run target quality (>=0.85) experiment")
@click.option("--semantic-safety", is_flag=True, help="Run Section 27 semantic cache false-positive safety experiment")
@click.option("--freshness", is_flag=True, help="Run Section 28 freshness experiment")
@click.option("--memory-pressure", is_flag=True, help="Run Section 29 memory pressure experiment")
@click.option("--voi", is_flag=True, help="Run Phase 2 VOI evaluation")
def benchmark(ctx, phase: str, ablation: bool, memory_ablation: bool, tool_ablation: bool, tool_vs_model: bool, tool_failure: bool, budget_study: bool, speculation_ablation: bool, speculation_break_even: bool, speculation_workload: bool, target_exp: bool, semantic_safety: bool, freshness: bool, memory_pressure: bool, voi: bool):
    """Run comparative benchmark suite and ablations."""
    if ctx.invoked_subcommand is not None:
        return

    harness = BenchmarkHarness()

    if speculation_ablation:
        click.echo("\nRunning AEON Phase 5 Speculation Component Ablation Study...\n")
        results = harness.run_speculation_ablation_study()
        click.echo("=" * 105)
        click.echo(
            f"{'Ablation Configuration':<32} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Acceptance':<10} | {'Speedup':<8} | {'Quality/$':<11}"
        )
        click.echo("=" * 105)
        for name, d in results.items():
            lat_str = f"{d['mean_latency_ms']:.1f}ms"
            cost_str = f"${d['mean_cost_usd']:.5f}"
            acc_str = f"{d['mean_acceptance_rate']*100:.1f}%"
            speed_str = f"{d['mean_speedup']:.2f}x"
            click.echo(
                f"{name:<32} | {d['quality']*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {acc_str:>10} | {speed_str:>8} | {d['quality_per_dollar']:>11.0f}"
            )
        click.echo("=" * 105)
        click.echo()
        return

    if speculation_break_even:
        click.echo("\nRunning Phase 5 Economic Break-Even Analysis Across Models and Draft Lengths...\n")
        res = harness.run_speculation_break_even_experiment()
        click.echo(json.dumps(res, indent=2))
        return

    if speculation_workload:
        click.echo("\nRunning Phase 5 Generation Workload Breakdown (8 Modalities)...\n")
        res = harness.run_speculation_workload_breakdown()
        click.echo(json.dumps(res, indent=2))
        return

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

    if phase == "5":
        click.echo("\nRunning AEON Phase 5 Speculative & Token-Efficient Inference Benchmark Suite...\n")
        results = harness.run_phase5()
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


from aeon.benchmarks.cli import (
    benchmark_list,
    benchmark_run,
    benchmark_compare,
    benchmark_ablation,
    benchmark_sensitivity,
    benchmark_pareto,
    benchmark_reproducibility,
    benchmark_validate,
    benchmark_report,
    benchmark_export
)

benchmark.add_command(benchmark_list, "list")
benchmark.add_command(benchmark_run, "run")
benchmark.add_command(benchmark_compare, "compare")
benchmark.add_command(benchmark_ablation, "ablation")
benchmark.add_command(benchmark_sensitivity, "sensitivity")
benchmark.add_command(benchmark_pareto, "pareto")
benchmark.add_command(benchmark_reproducibility, "reproducibility")
benchmark.add_command(benchmark_validate, "validate")
benchmark.add_command(benchmark_report, "report")
benchmark.add_command(benchmark_export, "export")


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


# =========================================================================
# Phase 5 Speculation Commands (Section 20)
# =========================================================================

@cli.group()
def speculation():
    """AEON Speculative and Token-Efficient Inference Controller (Phase 5)."""
    pass


@speculation.command(name="stats")
def speculation_stats():
    """Display online acceptance statistics and learned strategy profiles."""
    from aeon.speculation.controller import global_speculation_controller
    stats = global_speculation_controller.estimator.get_summary()
    click.echo("\n==================== AEON SPECULATION STATS ====================")
    click.echo(f"Global Acceptance EMA:        {stats['global_ema']*100:.1f}%")
    click.echo(f"Overall Acceptance Rate:      {stats['overall_acceptance_rate']*100:.1f}%")
    click.echo(f"Rounds Tracked:               {stats['rounds_tracked']}")
    click.echo("\nPer-Strategy Acceptance EMA:")
    for strat, val in stats['strategy_ema'].items():
        click.echo(f"  - {strat:<20}: {val*100:.1f}%")
    click.echo("\nPer-Draft-Length Acceptance EMA:")
    for length, val in sorted(stats['draft_length_ema'].items(), key=lambda kv: int(kv[0])):
        click.echo(f"  - Length {length:<3}: {val*100:.1f}%")
    click.echo("================================================================\n")


@speculation.command(name="benchmark")
def speculation_benchmark():
    """Run dedicated Phase 5 speculative decoding benchmark suite."""
    harness = BenchmarkHarness()
    click.echo("\nRunning AEON Phase 5 Speculative & Token-Efficient Inference Benchmark Suite...\n")
    results = harness.run_phase5()
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


@speculation.command(name="compare")
def speculation_compare():
    """Compare performance across all 8 generation workload categories."""
    harness = BenchmarkHarness()
    click.echo("\nRunning Phase 5 Generation Workload Breakdown (8 Modalities)...\n")
    res = harness.run_speculation_workload_breakdown()
    click.echo("=" * 95)
    click.echo(
        f"{'Workload Modality':<24} | {'Standard':<10} | {'Always-On':<10} | {'Adaptive':<10} | {'Speedup':<8} | {'Selected'}"
    )
    click.echo("=" * 95)
    for cat, d in res.items():
        click.echo(
            f"{cat:<24} | {d['standard_latency_ms']:>8.1f}ms | {d['always_on_latency_ms']:>8.1f}ms | {d['aeon_adaptive_latency_ms']:>8.1f}ms | {d['speedup_vs_standard']:>7.2f}x | {d['preferred_strategy']}"
        )
    click.echo("=" * 95)
    click.echo()


@speculation.command(name="config")
def speculation_config():
    """Inspect active speculative decoding configuration."""
    from aeon.speculation.controller import global_speculation_controller
    cfg = global_speculation_controller.config
    dump_fn = getattr(cfg, "model_dump", cfg.dict)
    click.echo(json.dumps(dump_fn(), indent=2))


@speculation.command(name="break-even")
def speculation_break_even():
    """Run theoretical economic break-even experiment across models and draft lengths."""
    harness = BenchmarkHarness()
    res = harness.run_speculation_break_even_experiment()
    click.echo(json.dumps(res, indent=2))


# =========================================================================
# Phase 6 Adaptive Reasoning Commands
# =========================================================================

@cli.group()
def reasoning():
    """AEON Adaptive Latent & Token-Efficient Reasoning Controller (Phase 6)."""
    pass


@reasoning.command(name="benchmark")
def reasoning_benchmark():
    """Run dedicated Phase 6 adaptive reasoning benchmark suite."""
    harness = BenchmarkHarness()
    click.echo("\nRunning AEON Phase 6 Adaptive Reasoning Benchmark Suite...\n")
    results = harness.run_phase6()
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


@reasoning.command(name="compare")
def reasoning_compare():
    """Compare reasoning performance across all 6 workload modalities."""
    harness = BenchmarkHarness()
    click.echo("\nRunning Phase 6 Reasoning Workload Breakdown (6 Modalities)...\n")
    res = harness.run_reasoning_workload_breakdown()
    click.echo("=" * 105)
    click.echo(
        f"{'Modality':<22} | {'Direct Latency':<15} | {'Heavy Cost':<14} | {'Adaptive Cost':<15} | {'Savings':<9} | {'Selected'}"
    )
    click.echo("=" * 105)
    for cat, d in res.items():
        click.echo(
            f"{cat:<22} | {d['mean_no_reason_latency_ms']:>12.1f}ms | ${d['mean_heavy_cost_usd']:>12.6f} | ${d['mean_adaptive_cost_usd']:>13.6f} | {d['cost_savings_vs_heavy']:>7.1f}% | {d['preferred_strategy']}"
        )
    click.echo("=" * 105)
    click.echo()


@reasoning.command(name="ablation")
def reasoning_ablation():
    """Run Phase 6 component ablation study."""
    harness = BenchmarkHarness()
    click.echo("\nRunning Phase 6 Reasoning Component Ablation Study...\n")
    res = harness.run_reasoning_ablation_study()
    click.echo("=" * 105)
    click.echo(
        f"{'Ablation Variant':<30} | {'Quality':<8} | {'Latency':<10} | {'Cost':<12} | {'Tokens':<8} | {'Quality/$':<11}"
    )
    click.echo("=" * 105)
    for name, d in res.items():
        click.echo(
            f"{name:<30} | {d['quality']*100:>6.1f}% | {d['mean_latency_ms']:>8.1f}ms | ${d['mean_cost_usd']:>10.6f} | {d['mean_reasoning_tokens']:>8.1f} | {d['quality_per_dollar']:>11.0f}"
        )
    click.echo("=" * 105)
    click.echo()


@reasoning.command(name="config")
def reasoning_config():
    """Inspect active adaptive reasoning configuration."""
    from aeon.reasoning.controller import global_reasoning_controller
    cfg = global_reasoning_controller.config
    dump_fn = getattr(cfg, "model_dump", cfg.dict)
    click.echo(json.dumps(dump_fn(), indent=2))


@cli.group(name="depth")
def depth():
    """AEON Adaptive Model Depth & Mixture-of-Depths (Phase 7)."""
    pass


@depth.command(name="benchmark")
@click.option("--runs", default=1, type=int, help="Number of evaluation passes")
def depth_benchmark(runs: int):
    """Run dedicated Phase 7 adaptive depth & MoD benchmark suite (7 Baselines)."""
    harness = BenchmarkHarness()
    click.echo("\nRunning AEON Phase 7 Adaptive Depth & MoD Benchmark Suite...\n")
    results = harness.run_phase7(runs=runs)
    click.echo("=" * 115)
    click.echo(
        f"{'Baseline Configuration':<36} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Tokens':<8} | {'Quality/$':<11} | {'Quality/TFLOP':<13}"
    )
    click.echo("=" * 115)
    for name, m in results.items():
        lat_str = f"{m.mean_latency_ms:.1f}ms"
        cost_str = f"${m.total_cost_usd:.5f}"
        click.echo(
            f"{name:<36} | {m.accuracy*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {m.total_tokens:>8} | {m.quality_per_dollar:>11.0f} | {m.quality_per_tflops:>13.4f}"
        )
    click.echo("=" * 115)
    click.echo()


@depth.command(name="compare")
def depth_compare():
    """Compare depth & compute across all 12 workload modalities."""
    harness = BenchmarkHarness()
    click.echo("\nRunning Phase 7 Depth Workload Breakdown (12 Modalities)...\n")
    res = harness.run_depth_workload_breakdown()
    click.echo("=" * 115)
    click.echo(
        f"{'Modality':<20} | {'Full Depth':<10} | {'Adapt Depth':<11} | {'Depth Red.':<10} | {'Full Cost':<12} | {'Adapt Cost':<12} | {'Exit %':<7}"
    )
    click.echo("=" * 115)
    for cat, d in res.items():
        click.echo(
            f"{cat:<20} | {d['mean_full_depth']:>10.1f} | {d['mean_adaptive_depth']:>11.1f} | {d['depth_reduction_pct']:>9.1f}% | ${d['mean_full_cost_usd']:>10.6f} | ${d['mean_adaptive_cost_usd']:>10.6f} | {d['early_exit_rate_pct']:>6.1f}%"
        )
    click.echo("=" * 115)
    click.echo()


@depth.command(name="ablation")
def depth_ablation():
    """Run Phase 7 depth and MoD component ablation study."""
    harness = BenchmarkHarness()
    click.echo("\nRunning Phase 7 Depth Component Ablation Study (10 Conditions)...\n")
    res = harness.run_depth_ablation_study()
    click.echo("=" * 110)
    click.echo(
        f"{'Ablation Variant':<30} | {'Quality':<8} | {'Latency':<10} | {'Cost':<12} | {'Mean Depth':<10} | {'Quality/$':<11}"
    )
    click.echo("=" * 110)
    for name, d in res.items():
        click.echo(
            f"{name:<30} | {d['quality']*100:>6.1f}% | {d['mean_latency_ms']:>8.1f}ms | ${d['mean_cost_usd']:>10.6f} | {d['mean_depth']:>10.1f} | {d['quality_per_dollar']:>11.0f}"
        )
    click.echo("=" * 110)
    click.echo()


@depth.command(name="config")
def depth_config():
    """Inspect active adaptive model depth and MoD configuration."""
    from aeon.depth.controller import global_depth_controller
    cfg = global_depth_controller.config
    dump_fn = getattr(cfg, "model_dump", cfg.dict)
    click.echo(json.dumps(dump_fn(), indent=2))


@depth.command(name="break-even")
def depth_break_even():
    """Compute and display Phase 7 economic break-even points."""
    harness = BenchmarkHarness()
    click.echo("\nComputing AEON Phase 7 Economic Break-Even Analysis...\n")
    be = harness.run_depth_break_even_analysis()
    click.echo(json.dumps(be, indent=2))


@depth.command(name="overhead")
def depth_overhead():
    """Measure AEON controller overhead for depth decision evaluation."""
    harness = BenchmarkHarness()
    click.echo("\nMeasuring Phase 7 Depth Controller Latency Overhead...\n")
    ov = harness.run_depth_controller_overhead()
    click.echo(json.dumps(ov, indent=2))


# =========================================================================
# Phase 8 Hardware & Quantization CLI Subsystem
# =========================================================================

@cli.group(name="hardware")
def hardware():
    """AEON Hardware Detection, Precision & Quantization Subsystem (Phase 8)."""
    pass


@hardware.command(name="info")
def hardware_info():
    """Display comprehensive host hardware and accelerator capability overview."""
    from aeon.hardware.controller import global_hardware_controller
    prof = global_hardware_controller.profile

    click.echo("\n==================== AEON HARDWARE PROFILE ====================")
    click.echo(f"Profile ID:          {prof.profile_id}")
    click.echo(f"Operating System:    {prof.os_name} ({prof.machine_arch})")
    click.echo(f"Host CPU:            {prof.cpu_model}")
    click.echo(f"Physical/Log Cores:  {prof.cpu_cores_physical} physical / {prof.cpu_cores_logical} logical")
    click.echo(f"System RAM:          {prof.ram_total_mb:.1f} MB (Available: {prof.ram_available_mb:.1f} MB)")
    click.echo(f"Unified Memory:      {prof.unified_memory}")
    click.echo(f"Memory Pressure:     {prof.memory_pressure.value.upper()}")
    click.echo(f"Thermal State:       {prof.thermal_pressure.value.upper()}")
    click.echo("-------------------- ACCELERATOR ENGINE --------------------")
    click.echo(f"Device Class:        {prof.accelerator_type.value.upper()}")
    click.echo(f"Accelerator Name:    {prof.accelerator_name}")
    click.echo(f"Device Count:        {prof.accelerator_count}")
    click.echo(f"VRAM / Addr Memory:  {prof.accelerator_memory_mb:.1f} MB")
    click.echo(f"PyTorch Version:     {prof.pytorch_version} (MPS: {prof.mps_available}, CUDA: {prof.cuda_available})")
    click.echo("------------------- PRECISION COMPATIBILITY ----------------")
    for p, status in prof.supported_precisions.items():
        click.echo(f"  - {p:<8}: {status.upper()}")
    click.echo("------------------- QUANTIZATION STATUS -------------------")
    for q, status in prof.quantization_support.items():
        click.echo(f"  - {q:<18}: {status.upper()}")
    click.echo("===============================================================\n")


@hardware.command(name="detect")
def hardware_detect():
    """Detect and output raw machine hardware profile as JSON."""
    from aeon.hardware.detector import HardwareDetector
    detector = HardwareDetector()
    prof = detector.detect_hardware(force_refresh=True)
    dump_fn = getattr(prof, "model_dump", prof.dict)
    click.echo(json.dumps(dump_fn(), indent=2))


@hardware.command(name="precision")
def hardware_precision():
    """Inspect standard numerical precision profiles and analytical scaling factors."""
    from aeon.hardware.precision import STANDARD_PRECISION_PROFILES
    click.echo("\n==================== AEON PRECISION PROFILES ====================")
    click.echo(f"{'Precision':<10} | {'Bits':<6} | {'Mem Factor':<12} | {'Compute Factor':<15} | {'Quality Delta':<14} | {'Status':<12}")
    click.echo("=" * 80)
    for p, prof in STANDARD_PRECISION_PROFILES.items():
        click.echo(
            f"{p.value:<10} | {prof.bits:<6} | {prof.memory_factor:<12.2f} | {prof.compute_factor:<15.2f} | {prof.quality_delta:<14.4f} | {prof.status.value.upper():<12}"
        )
    click.echo("=" * 80)
    click.echo()


@hardware.command(name="quantization")
def hardware_quantization():
    """Inspect quantization methods, memory reductions, and real vs simulated status."""
    from aeon.hardware.controller import global_hardware_controller
    prof = global_hardware_controller.profile
    click.echo("\n==================== AEON QUANTIZATION METHODS ====================")
    click.echo(f"{'Method':<20} | {'Status':<14} | {'Classification':<16} | {'Target Precision':<16}")
    click.echo("=" * 72)
    for q, status in prof.quantization_support.items():
        is_sim = "SIMULATED" if "simulated" in q or status == "simulated" else "SUPPORTED"
        prec = "INT8" if "int8" in q else "FP8"
        click.echo(f"{q:<20} | {status.upper():<14} | {is_sim:<16} | {prec:<16}")
    click.echo("=" * 72)
    click.echo()


@hardware.command(name="benchmark")
@click.option("--runs", default=1, type=int, help="Number of evaluation passes")
def hardware_benchmark(runs: int):
    """Run dedicated Phase 8 hardware & quantization benchmark suite (7 Baselines)."""
    harness = BenchmarkHarness()
    click.echo("\nRunning AEON Phase 8 Hardware & Quantization Benchmark Suite...\n")
    results = harness.run_phase8(runs=runs)
    click.echo("=" * 115)
    click.echo(
        f"{'Baseline Configuration':<38} | {'Quality':<7} | {'Latency':<9} | {'Cost ($)':<10} | {'Tokens':<8} | {'Quality/$':<11} | {'Quality/TFLOP':<13}"
    )
    click.echo("=" * 115)
    for name, m in results.items():
        lat_str = f"{m.mean_latency_ms:.1f}ms"
        cost_str = f"${m.total_cost_usd:.5f}"
        click.echo(
            f"{name:<38} | {m.accuracy*100:>6.1f}% | {lat_str:>9} | {cost_str:>10} | {m.total_tokens:>8} | {m.quality_per_dollar:>11.0f} | {m.quality_per_tflops:>13.4f}"
        )
    click.echo("=" * 115)
    click.echo()


@hardware.command(name="ablation")
def hardware_ablation():
    """Run Phase 8 hardware, precision, and quantization ablation study (10 Conditions)."""
    harness = BenchmarkHarness()
    click.echo("\nRunning Phase 8 Hardware & Quantization Ablation Study (10 Conditions)...\n")
    res = harness.run_hardware_ablation_study()
    click.echo("=" * 105)
    click.echo(
        f"{'Ablation Variant':<30} | {'Quality':<8} | {'Latency':<10} | {'Cost ($/req)':<14} | {'Quality/$':<11}"
    )
    click.echo("=" * 105)
    for name, d in res.items():
        click.echo(
            f"{name:<30} | {d['quality']*100:>6.1f}% | {d['mean_latency_ms']:>8.1f}ms | ${d['mean_cost_usd']:>12.6f} | {d['quality_per_dollar']:>11.0f}"
        )
    click.echo("=" * 105)
    click.echo()


@hardware.command(name="config")
def hardware_config():
    """Inspect active hardware controller configuration."""
    from aeon.hardware.controller import global_hardware_controller
    cfg = global_hardware_controller.config
    dump_fn = getattr(cfg, "model_dump", cfg.dict)
    click.echo(json.dumps(dump_fn(), indent=2))


@hardware.command(name="break-even")
def hardware_break_even():
    """Compute and display Phase 8 economic break-even points."""
    harness = BenchmarkHarness()
    click.echo("\nComputing AEON Phase 8 Economic Break-Even Analysis...\n")
    be = harness.run_hardware_break_even_analysis()
    click.echo(json.dumps(be, indent=2))


@hardware.command(name="overhead")
def hardware_overhead():
    """Measure AEON controller overhead for hardware & precision decision evaluation."""
    harness = BenchmarkHarness()
    click.echo("\nMeasuring Phase 8 Hardware Controller Latency Overhead...\n")
    ov = harness.run_hardware_controller_overhead(iterations=100)
    click.echo(json.dumps(ov, indent=2))


if __name__ == "__main__":
    cli()


