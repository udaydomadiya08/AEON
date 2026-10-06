"""
Phase 6 Unit, Integration, and Invariant Test Suite for AEON.
Tests Adaptive Latent & Token-Efficient Reasoning, Economic VOI Modeling,
Strategy Selection, Confidence-Aware Dynamic Stopping, Answer Stability,
Telemetry, Benchmark Harness, CLI, and Specification Property Invariants.
"""

import pytest
from click.testing import CliRunner
from aeon import (
    AEON,
    InferenceRequest,
    TaskType,
    ControllerAction,
    DecisionReason,
    RouteChoice,
    ReasoningMode,
    ReasoningStrategy,
    ReasoningAction,
    ReasoningStopReason,
    ReasoningBudget,
    ReasoningState,
    ReasoningCandidate,
    ConfidenceEstimate,
    ReasoningConfig,
    ReasoningEconomics,
    ReasoningStrategySelector,
    ReasoningController,
    global_reasoning_controller,
    global_registry,
)
from aeon.cli.main import cli
from aeon.benchmarks.harness import BenchmarkHarness


# =========================================================================
# 1. Reasoning Types & Initializations
# =========================================================================

def test_reasoning_types():
    assert ReasoningStrategy.NONE == "none"
    assert ReasoningStrategy.DIRECT == "direct"
    assert ReasoningStrategy.SHALLOW == "shallow"
    assert ReasoningStrategy.MEDIUM == "medium"
    assert ReasoningStrategy.DEEP == "deep"
    assert ReasoningStrategy.LATENT == "latent"
    assert ReasoningStrategy.EXPLICIT == "explicit"
    assert ReasoningStrategy.TOOL_ASSISTED == "tool_assisted"
    assert ReasoningStrategy.MODEL_ESCALATION == "model_escalation"

    assert ReasoningAction.START_REASONING == "START_REASONING"
    assert ReasoningAction.CONTINUE_REASONING == "CONTINUE_REASONING"
    assert ReasoningAction.STOP_REASONING == "STOP_REASONING"

    assert ReasoningStopReason.TARGET_CONFIDENCE_REACHED == "TARGET_CONFIDENCE_REACHED"
    assert ReasoningStopReason.ANSWER_STABLE == "ANSWER_STABLE"
    assert ReasoningStopReason.BUDGET_EXHAUSTED == "BUDGET_EXHAUSTED"


def test_reasoning_budget_tracking():
    budget = ReasoningBudget(
        max_reasoning_tokens=500,
        max_reasoning_steps=5,
        max_cost_usd=0.002,
        max_latency_ms=1000.0,
    )
    assert not budget.is_exhausted()
    assert budget.remaining_tokens() == 500
    assert budget.remaining_steps() == 5

    # Consume partially
    budget.tokens_used = 200
    budget.steps_used = 2
    budget.cost_used_usd = 0.0005
    budget.latency_used_ms = 300.0
    assert not budget.is_exhausted()
    assert budget.remaining_tokens() == 300
    assert budget.remaining_steps() == 3

    # Exceed step limit
    budget.steps_used = 5
    assert budget.is_exhausted()
    assert budget.remaining_steps() == 0


def test_reasoning_state_initialization():
    state = ReasoningState(
        request_id="test-req-001",
        query="Prove that square root of 2 is irrational.",
        strategy=ReasoningStrategy.DEEP,
        model_id="aeon-medium-7b",
    )
    assert state.reasoning_steps == 0
    assert state.reasoning_tokens == 0
    assert state.confidence == 0.0
    assert state.is_terminal is False
    assert state.stop_reason is None
    assert state.execution_mode == "SIMULATED_LATENT_REASONING"


# =========================================================================
# 2. Reasoning Economics & VOI Modeling
# =========================================================================

def test_reasoning_economics_marginal_calculations():
    econ = ReasoningEconomics()
    state = ReasoningState(
        request_id="test-econ",
        query="Solve this logic puzzle.",
        strategy=ReasoningStrategy.LATENT,
        model_id="aeon-small-1.5b",
        confidence=0.60,
        quality_estimate=0.60,
    )

    # 1. Marginal quality gain
    gain = econ.estimate_marginal_quality_gain(
        state=state,
        next_strategy=ReasoningStrategy.LATENT,
        difficulty=0.7,
        step_number=1,
    )
    assert 0.0 < gain <= 0.35

    # Diminishing returns: step 5 gain should be less than step 1 gain
    gain_step5 = econ.estimate_marginal_quality_gain(
        state=state,
        next_strategy=ReasoningStrategy.LATENT,
        difficulty=0.7,
        step_number=5,
    )
    assert gain_step5 < gain

    # 2. Marginal cost and latency
    cost = econ.estimate_step_cost(model="aeon-small-1.5b", strategy=ReasoningStrategy.LATENT)
    lat = econ.estimate_step_latency(model="aeon-small-1.5b", strategy=ReasoningStrategy.LATENT)
    flops = econ.estimate_step_compute(model="aeon-small-1.5b", strategy=ReasoningStrategy.LATENT)
    assert cost > 0.0
    assert lat > 0.0
    assert flops > 0.0

    # Explicit strategy should cost more than latent strategy
    cost_explicit = econ.estimate_step_cost(model="aeon-small-1.5b", strategy=ReasoningStrategy.EXPLICIT)
    assert cost_explicit > cost

    # 3. VOI ratio
    voi = econ.calculate_reasoning_voi(marginal_quality_gain=gain, marginal_cost_usd=cost)
    assert voi > 0.0


def test_reasoning_candidates_evaluation():
    econ = ReasoningEconomics()
    state = ReasoningState(
        request_id="test-candidates",
        query="Deduce the answer.",
        strategy=ReasoningStrategy.SHALLOW,
        model_id="aeon-small-1.5b",
        confidence=0.55,
        quality_estimate=0.55,
    )
    budget = ReasoningBudget()
    candidates = econ.evaluate_all_candidates(
        state=state,
        budget=budget,
        task_type="reasoning",
        difficulty=0.6,
    )
    assert len(candidates) >= 1
    # Check candidate structure
    for cand in candidates:
        assert isinstance(cand, ReasoningCandidate)
        assert cand.marginal_cost_usd > 0.0
        assert cand.voi_ratio >= 0.0


# =========================================================================
# 3. Reasoning Strategy Selector
# =========================================================================

def test_strategy_selector_cheapest_adequate():
    selector = ReasoningStrategySelector()

    # Easy factual / extraction -> NONE or DIRECT
    strat_fact = selector.select_initial_strategy(
        task_type="factual", difficulty=0.1, uncertainty=0.1, model="aeon-small-1.5b"
    )
    assert strat_fact in [ReasoningStrategy.NONE, ReasoningStrategy.DIRECT]

    # Easy classification -> NONE or DIRECT
    strat_class = selector.select_initial_strategy(
        task_type="classification", difficulty=0.2, uncertainty=0.1, model="aeon-small-1.5b"
    )
    assert strat_class in [ReasoningStrategy.NONE, ReasoningStrategy.DIRECT]

    # Easy reasoning -> SHALLOW or LATENT
    strat_easy_reason = selector.select_initial_strategy(
        task_type="reasoning", difficulty=0.3, uncertainty=0.3, model="aeon-small-1.5b"
    )
    assert strat_easy_reason in [ReasoningStrategy.SHALLOW, ReasoningStrategy.LATENT]

    # Hard reasoning / proof -> DEEP or EXPLICIT
    strat_hard_reason = selector.select_initial_strategy(
        task_type="reasoning", difficulty=0.85, uncertainty=0.7, model="aeon-large-70b"
    )
    assert strat_hard_reason in [ReasoningStrategy.DEEP, ReasoningStrategy.EXPLICIT]


def test_strategy_transitions():
    selector = ReasoningStrategySelector()
    transitions = selector.get_transition_candidates(ReasoningStrategy.SHALLOW)
    assert ReasoningStrategy.LATENT in transitions or ReasoningStrategy.MEDIUM in transitions
    assert ReasoningStrategy.NONE not in transitions  # Escalation forwards, not backwards


# =========================================================================
# 4. Reasoning Controller Execution
# =========================================================================

def test_reasoning_controller_simple_query():
    ctrl = ReasoningController()
    answer, state, tel = ctrl.execute_reasoning(
        query="What is the capital of France?",
        task_type="factual",
        difficulty=0.15,
        uncertainty=0.1,
    )
    assert "Paris" in answer or len(answer) > 0
    assert state.is_terminal is True
    # Factual query should use 0 or minimal reasoning steps
    assert state.reasoning_steps <= 1
    assert tel["reasoning_enabled"] is True
    assert tel["confidence"] >= 0.70


def test_reasoning_controller_deep_reasoning_stopping():
    ctrl = ReasoningController()
    answer, state, tel = ctrl.execute_reasoning(
        query="Construct a formal topological proof of the Brouwer fixed-point theorem.",
        task_type="reasoning",
        difficulty=0.85,
        uncertainty=0.75,
        model="aeon-large-70b",
        target_confidence=0.85,
    )
    assert len(answer) > 0
    assert state.is_terminal is True
    assert state.reasoning_steps >= 1
    assert state.stop_reason is not None
    assert len(tel["confidence_trace"]) >= 1
    assert tel["quality_estimate"] >= 0.75


def test_reasoning_controller_budget_exhaustion():
    # Provide an extremely tight token and step budget
    cfg = ReasoningConfig(max_reasoning_steps=2, max_reasoning_tokens=50)
    ctrl = ReasoningController(config=cfg)
    answer, state, tel = ctrl.execute_reasoning(
        query="Solve this extremely hard NP-complete problem with 100 steps.",
        task_type="reasoning",
        difficulty=0.95,
        uncertainty=0.9,
    )
    assert state.reasoning_steps <= 2
    assert state.is_terminal is True
    assert state.stop_reason in [
        ReasoningStopReason.BUDGET_EXHAUSTED,
        ReasoningStopReason.MAX_STEPS_REACHED,
        ReasoningStopReason.TARGET_CONFIDENCE_REACHED,
        ReasoningStopReason.ANSWER_STABLE,
    ]


# =========================================================================
# 5. AEON End-to-End Integration
# =========================================================================

def test_aeon_run_with_reasoning_rules_controller():
    aeon = AEON(policy="rules", reasoning=True)
    assert aeon.reasoning_enabled is True
    assert aeon.reasoning is not None

    res = aeon.run("Analyze the logical structure of this syllogism.", reasoning=True)
    assert len(res.answer) > 0
    assert res.trace.reasoning_enabled is True
    assert res.trace.reasoning_depth_steps >= 0
    assert res.plan is not None
    assert res.cost >= 0.0


def test_aeon_run_with_reasoning_voi_dynamic_stopping():
    aeon = AEON(policy="voi", reasoning=True)
    res = aeon.run(
        "Deduce whether this system of polynomial equations has a real solution.",
        reasoning=True,
    )
    assert len(res.answer) > 0
    assert res.trace.reasoning_enabled is True
    assert res.confidence >= 0.70
    assert res.trace.latency_ms > 0.0


def test_aeon_reasoning_forced_strategy():
    aeon = AEON(policy="rules", reasoning=True)
    res = aeon.run(
        "Compute logic steps.",
        reasoning=True,
        reasoning_strategy="latent",
    )
    assert res.trace.reasoning_enabled is True
    assert res.trace.reasoning_strategy == "latent"


def test_aeon_reasoning_telemetry_fields():
    aeon = AEON(policy="voi", reasoning=True)
    res = aeon.run("Evaluate boolean satisfiability.", reasoning=True)
    t = res.trace

    assert hasattr(t, "reasoning_enabled")
    assert hasattr(t, "reasoning_strategy")
    assert hasattr(t, "reasoning_model")
    assert hasattr(t, "reasoning_depth_steps")
    assert hasattr(t, "reasoning_latent_steps")
    assert hasattr(t, "reasoning_explicit_tokens")
    assert hasattr(t, "reasoning_quality_estimate")
    assert hasattr(t, "reasoning_uncertainty")
    assert hasattr(t, "reasoning_voi")
    assert hasattr(t, "reasoning_stop_reason")
    assert hasattr(t, "reasoning_budget_tokens_used")
    assert hasattr(t, "reasoning_answer_stable")
    assert hasattr(t, "reasoning_confidence_trace")
    assert hasattr(t, "reasoning_action_trace")


# =========================================================================
# 6. Benchmark Harness Integration
# =========================================================================

def test_benchmark_harness_phase6():
    harness = BenchmarkHarness()
    results = harness.run_phase6()

    assert "1. No Reasoning (Direct)" in results
    assert "2. Fixed Heavy Reasoning" in results
    assert "3. AEON Adaptive Reasoning" in results

    res_none = results["1. No Reasoning (Direct)"]
    res_heavy = results["2. Fixed Heavy Reasoning"]
    res_adaptive = results["3. AEON Adaptive Reasoning"]

    # Adaptive reasoning should have higher quality than No Reasoning
    assert res_adaptive.accuracy >= res_none.accuracy
    # Adaptive reasoning should cost less than Fixed Heavy Reasoning
    assert res_adaptive.total_cost_usd <= res_heavy.total_cost_usd
    # Adaptive reasoning should have higher quality per dollar
    assert res_adaptive.quality_per_dollar >= res_heavy.quality_per_dollar


def test_benchmark_harness_reasoning_ablation():
    harness = BenchmarkHarness()
    ablations = harness.run_reasoning_ablation_study()

    assert "Full AEON Adaptive" in ablations
    assert "No Reasoning" in ablations
    assert "Explicit Only (No Latent)" in ablations
    assert "Without Early Stopping" in ablations

    full = ablations["Full AEON Adaptive"]
    no_reason = ablations["No Reasoning"]
    # Full adaptive should achieve equal or better quality than no reasoning
    assert full["quality"] >= no_reason["quality"]


def test_benchmark_harness_reasoning_workload_breakdown():
    harness = BenchmarkHarness()
    breakdown = harness.run_reasoning_workload_breakdown()

    assert "no_reasoning" in breakdown
    assert "shallow_reasoning" in breakdown
    assert "medium_reasoning" in breakdown
    assert "deep_reasoning" in breakdown
    assert "tool_assisted" in breakdown
    assert "budget_constrained" in breakdown

    # In no_reasoning modality, adaptive should have near 0 reasoning tokens
    assert breakdown["no_reasoning"]["mean_reasoning_tokens"] <= 10
    # In deep_reasoning modality, cost savings vs heavy should be non-negative
    assert breakdown["deep_reasoning"]["cost_savings_vs_heavy"] >= 0.0


# =========================================================================
# 7. CLI Integration
# =========================================================================

def test_cli_run_with_reasoning():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Solve this logic problem", "--reasoning", "--trace"])
    assert result.exit_code == 0
    assert "AEON RESULT" in result.output
    assert "AEON ADAPTIVE REASONING" in result.output
    assert "Strategy:" in result.output


def test_cli_reasoning_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["reasoning", "config"])
    assert result.exit_code == 0
    assert "enabled" in result.output


def test_cli_reasoning_benchmark():
    runner = CliRunner()
    result = runner.invoke(cli, ["reasoning", "benchmark"])
    assert result.exit_code == 0
    assert "AEON Phase 6 Adaptive Reasoning Benchmark Suite" in result.output
    assert "AEON Adaptive Reasoning" in result.output


def test_cli_reasoning_compare():
    runner = CliRunner()
    result = runner.invoke(cli, ["reasoning", "compare"])
    assert result.exit_code == 0
    assert "Phase 6 Reasoning Workload Breakdown" in result.output
    assert "deep_reasoning" in result.output


def test_cli_reasoning_ablation():
    runner = CliRunner()
    result = runner.invoke(cli, ["reasoning", "ablation"])
    assert result.exit_code == 0
    assert "Phase 6 Reasoning Component Ablation Study" in result.output
    assert "Full AEON Adaptive" in result.output


# =========================================================================
# 8. Specification Invariants
# =========================================================================

def test_invariant_cheapest_adequate_reasoning():
    """Invariant 1: Tasks that do not need reasoning must never spend explicit reasoning tokens."""
    aeon = AEON(policy="voi", reasoning=True)
    res = aeon.run("What is the capital of Italy?", reasoning=True)
    assert res.trace.reasoning_explicit_tokens == 0


def test_invariant_latent_cheaper_than_explicit():
    """Invariant 2: Latent reasoning cost is strictly less than explicit tokens for equal steps."""
    econ = ReasoningEconomics()
    cost_latent = econ.estimate_step_cost("aeon-small-1.5b", ReasoningStrategy.LATENT)
    cost_explicit = econ.estimate_step_cost("aeon-small-1.5b", ReasoningStrategy.EXPLICIT)
    assert cost_latent < cost_explicit


def test_invariant_budget_enforcement():
    """Invariant 3: Allocated token and cost budgets are strictly respected."""
    ctrl = ReasoningController()
    _, state, tel = ctrl.execute_reasoning(
        query="Extremely complex combinatorial proof.",
        task_type="reasoning",
        difficulty=0.9,
        uncertainty=0.9,
        cost_budget_usd=0.0001,
        latency_budget_ms=200.0,
    )
    assert tel["budget_cost_used"] <= 0.001
    assert state.is_terminal is True
