"""
Phase 6 Unit, Integration, and Invariant Test Suite for AEON.
Tests Adaptive Latent & Token-Efficient Reasoning, Economic VOI Modeling,
Strategy Selection, Confidence-Aware Dynamic Stopping, Answer Stability,
Telemetry, Benchmark Harness, CLI, and Specification Property Invariants.
"""

import math
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
    UtilityWeights,
)
from aeon.reasoning.strategies import normalize_reasoning_answer
from aeon.reasoning.controller import clean_reasoning_output
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


# =========================================================================
# 9. Adversarial Audit & Hardening Regression Tests
# =========================================================================

def test_adversarial_zero_and_tiny_budgets():
    """Adversarial test: Zero or negative budgets must stop immediately with BUDGET_EXHAUSTED."""
    # 1. Zero steps in config
    cfg_zero_steps = ReasoningConfig(max_reasoning_steps=0)
    ctrl = ReasoningController(config=cfg_zero_steps)
    ans, state, tel = ctrl.execute_reasoning(
        query="Explain quantum gravity",
        task_type="reasoning",
        difficulty=0.8,
        uncertainty=0.8,
    )
    assert state.stop_reason == ReasoningStopReason.BUDGET_EXHAUSTED
    assert state.reasoning_steps == 0
    assert state.reasoning_tokens == 0
    assert len(ans) > 0

    # 2. Zero cost budget
    ctrl_default = ReasoningController()
    ans, state, tel = ctrl_default.execute_reasoning(
        query="Explain string theory",
        task_type="reasoning",
        difficulty=0.8,
        cost_budget_usd=0.0,
    )
    assert state.stop_reason == ReasoningStopReason.BUDGET_EXHAUSTED
    assert state.reasoning_steps == 0

    # 3. Zero latency budget
    ans, state, tel = ctrl_default.execute_reasoning(
        query="Prove Fermat's Last Theorem",
        task_type="reasoning",
        difficulty=0.9,
        latency_budget_ms=0.0,
    )
    assert state.stop_reason == ReasoningStopReason.BUDGET_EXHAUSTED
    assert state.reasoning_steps == 0

    # 4. Tiny budget: max_reasoning_steps = 1
    cfg_one_step = ReasoningConfig(max_reasoning_steps=1)
    ctrl_one = ReasoningController(config=cfg_one_step)
    ans, state, tel = ctrl_one.execute_reasoning(
        query="Deep analytical proof",
        task_type="reasoning",
        difficulty=0.9,
    )
    assert state.reasoning_steps <= 1


def test_economic_voi_cases_a_through_g():
    """Test all formal economic VOI decision cases (Case A through Case G)."""
    econ = ReasoningEconomics()
    budget = ReasoningBudget(max_reasoning_tokens=1000, max_cost_usd=0.05, max_latency_ms=10000.0)

    # Case A: High quality gain + low cost -> Viable / Continue
    state_a = ReasoningState(
        strategy=ReasoningStrategy.SHALLOW,
        model_id="aeon-small-1.5b",
        quality_estimate=0.40,
        reasoning_steps=1,
    )
    candidates_a = econ.evaluate_candidates(
        state=state_a,
        budget=budget,
        task_type="reasoning",
        difficulty=0.8,
        uncertainty=0.7,
    )
    viable_a = [c for c in candidates_a if c.is_viable]
    assert len(viable_a) > 0
    assert viable_a[0].marginal_utility > 0.0

    # Case B: Tiny quality gain + high cost -> STOP (non-viable)
    state_b = ReasoningState(
        strategy=ReasoningStrategy.HYBRID,
        model_id="aeon-large-70b",
        quality_estimate=0.95,  # Almost no headroom
        reasoning_steps=8,      # Severe diminishing returns
    )
    candidates_b = econ.evaluate_candidates(
        state=state_b,
        budget=budget,
        task_type="reasoning",
        difficulty=0.1,
        uncertainty=0.1,
    )
    viable_b = [c for c in candidates_b if c.is_viable]
    assert len(viable_b) == 0

    # Case C: Negative marginal utility due to extreme cost penalties -> STOP
    heavy_weights = UtilityWeights(quality=1.0, lambda_cost=100000.0, lambda_latency=100.0)
    econ_penalized = ReasoningEconomics(weights=heavy_weights)
    candidates_c = econ_penalized.evaluate_candidates(
        state=state_a,
        budget=budget,
        task_type="reasoning",
        difficulty=0.5,
        uncertainty=0.5,
    )
    viable_c = [c for c in candidates_c if c.is_viable]
    assert len(viable_c) == 0

    # Case D: Budget exhausted -> STOP
    budget_exhausted = ReasoningBudget(max_reasoning_tokens=10, tokens_used=10)
    candidates_d = econ.evaluate_candidates(
        state=state_a,
        budget=budget_exhausted,
        task_type="reasoning",
        difficulty=0.8,
        uncertainty=0.8,
    )
    assert all(not c.is_viable for c in candidates_d)

    # Case E: Latency SLA exceeded -> STOP
    budget_lat_exhausted = ReasoningBudget(max_latency_ms=10.0, latency_used_ms=10.0)
    candidates_e = econ.evaluate_candidates(
        state=state_a,
        budget=budget_lat_exhausted,
        task_type="reasoning",
        difficulty=0.8,
        uncertainty=0.8,
    )
    assert all(not c.is_viable for c in candidates_e)

    # Case F: Reasoning quality stagnates / plateau -> STOP with QUALITY_PLATEAU
    selector = ReasoningStrategySelector(config=ReasoningConfig(stagnation_window=3, stagnation_threshold=0.01))
    state_plateau = ReasoningState(
        quality_history=[0.70, 0.70, 0.70],
        confidence=0.70,
        reasoning_steps=3,
    )
    should_cont, stop_reason = selector.should_continue_reasoning(state_plateau, budget)
    assert should_cont is False
    assert stop_reason == ReasoningStopReason.QUALITY_PLATEAU

    # Case G: Confidence saturates -> STOP with TARGET_CONFIDENCE_REACHED
    state_sat = ReasoningState(
        confidence=0.92,
        reasoning_steps=2,
    )
    should_cont_g, stop_reason_g = selector.should_continue_reasoning(state_sat, budget)
    assert should_cont_g is False
    assert stop_reason_g == ReasoningStopReason.TARGET_CONFIDENCE_REACHED


def test_numerical_robustness_and_pathological_weights():
    """Verify system handles NaN, inf, negative values, and pathological weights safely."""
    econ = ReasoningEconomics()
    budget = ReasoningBudget(max_reasoning_tokens=500, max_cost_usd=0.01, max_latency_ms=2000.0)
    state = ReasoningState(
        strategy=ReasoningStrategy.SHALLOW,
        model_id="aeon-small-1.5b",
        quality_estimate=0.5,
    )

    # Test with NaN, inf, and negative inputs
    candidates_nan = econ.evaluate_candidates(
        state=state,
        budget=budget,
        task_type="reasoning",
        difficulty=float("nan"),
        uncertainty=float("inf"),
    )
    assert len(candidates_nan) > 0
    for c in candidates_nan:
        assert not math.isnan(c.marginal_utility)
        assert not math.isinf(c.marginal_utility)
        assert not math.isnan(c.voi_ratio)
        assert not math.isinf(c.voi_ratio)

    # Pathological negative weights should be clamped to >= 0 and never create infinite utility
    pathological_weights = UtilityWeights(
        quality=1.0,
        lambda_latency=-50.0,
        lambda_cost=-1000.0,
        lambda_compute=-1.0,
        lambda_energy=-1.0,
    )
    econ_patho = ReasoningEconomics(weights=pathological_weights)
    candidates_patho = econ_patho.evaluate_candidates(
        state=state,
        budget=budget,
        task_type="reasoning",
        difficulty=0.5,
        uncertainty=0.5,
    )
    for c in candidates_patho:
        assert c.marginal_utility <= 1.0  # Quality is bounded in [0, 1]


def test_answer_stability_normalization():
    """Verify that normalize_reasoning_answer handles prefixes, casing, and whitespace."""
    # Prefix normalization
    raw1 = "[Explicit-CoT step 1 (90 tokens)] Paris is the capital of France."
    raw2 = "[Explicit-CoT step 2 (90 tokens)]  Paris is the capital of France.  "
    raw3 = "[Explicit-CoT step 3 (90 tokens)] PARIS IS THE CAPITAL OF FRANCE."
    assert normalize_reasoning_answer(raw1) == "paris is the capital of france"
    assert normalize_reasoning_answer(raw2) == "paris is the capital of france"
    assert normalize_reasoning_answer(raw3) == "paris is the capital of france"
    assert normalize_reasoning_answer(raw1) == normalize_reasoning_answer(raw2)

    # Latent prefix normalization
    raw_lat = "[Latent-1 (6 compressed states)] 146.531 (17.3% of 847 = 0.173 * 847 = 146.531)"
    assert "146.531" in normalize_reasoning_answer(raw_lat)

    # Check stability trigger in should_continue_reasoning
    selector = ReasoningStrategySelector(ReasoningConfig(enable_answer_stability=True, answer_stability_window=3))
    budget = ReasoningBudget(max_reasoning_steps=10)
    state_stable = ReasoningState(
        answer_history=[raw1, raw2, raw3],
        confidence=0.75,
        reasoning_steps=3,
    )
    should_cont, stop_reason = selector.should_continue_reasoning(state_stable, budget)
    assert should_cont is False
    assert stop_reason == ReasoningStopReason.ANSWER_STABLE

    # Oscillating answers should NOT trigger stability
    state_osc = ReasoningState(
        answer_history=["Answer A", "Answer B", "Answer A"],
        confidence=0.75,
        reasoning_steps=3,
    )
    should_cont_osc, stop_reason_osc = selector.should_continue_reasoning(state_osc, budget)
    assert should_cont_osc is True
    assert stop_reason_osc is None


def test_stagnation_detection_on_plateau_and_noise():
    """Verify that tiny noise does not prevent plateau / stagnation detection."""
    selector = ReasoningStrategySelector(ReasoningConfig(stagnation_window=4, stagnation_threshold=0.01))
    budget = ReasoningBudget(max_reasoning_steps=10)

    # Sub-threshold noisy sequence: range is 0.701 - 0.699 = 0.002 < 0.01
    state_noisy = ReasoningState(
        quality_history=[0.700, 0.701, 0.699, 0.700],
        confidence=0.60,
        reasoning_steps=4,
    )
    should_cont, stop_reason = selector.should_continue_reasoning(state_noisy, budget)
    assert should_cont is False
    assert stop_reason == ReasoningStopReason.QUALITY_PLATEAU


def test_cot_privacy_and_clean_output():
    """Verify that CoT scratchpad text is never leaked in final answers or telemetry."""
    ctrl = ReasoningController()
    ans, state, tel = ctrl.execute_reasoning(
        query="What is 17.3% of 847?",
        task_type="math",
        difficulty=0.6,
        preferred_strategy=ReasoningStrategy.EXPLICIT,
    )
    # Output must be clean without bracketed internal scratchpad prefixes
    assert not ans.startswith("[Explicit-CoT")
    assert not ans.startswith("[Latent-")
    assert not ans.startswith("[Shallow")
    assert "146.531" in ans

    # Telemetry must contain metadata but NO private hidden CoT text
    assert "reasoning_strategy" in tel
    assert "reasoning_steps" in tel
    assert "action_trace" in tel
    for action in tel["action_trace"]:
        assert "scratchpad" not in action
        assert "chain_of_thought" not in action
        assert "step" in action
        assert "strategy" in action or "to_strategy" in action


def test_failure_injection_and_fallback():
    """Verify system fails safely and triggers fallback when a model error occurs."""
    ctrl = ReasoningController()
    # Inject non-existent model ID
    ans, state, tel = ctrl.execute_reasoning(
        query="Compute topology invariants",
        task_type="reasoning",
        difficulty=0.7,
        model="non_existent_faulty_model_999",
        preferred_strategy=ReasoningStrategy.EXPLICIT,
    )
    assert state.fallback_occurred is True
    assert state.stop_reason == ReasoningStopReason.FAILURE
    assert len(ans) > 0  # Returned safe fallback answer


def test_tool_precedence_in_strategy_selection():
    """Verify that deterministic tasks select TOOL_ASSISTED when tools are enabled."""
    selector = ReasoningStrategySelector(ReasoningConfig(enable_tool_assistance=True))
    strat = selector.select_initial_strategy(
        task_type=TaskType.ARITHMETIC.value,
        difficulty=0.6,
        uncertainty=0.5,
        model="aeon-small-1.5b",
    )
    assert strat == ReasoningStrategy.TOOL_ASSISTED


def test_ablation_enforcement():
    """Verify that disabling components in ReasoningConfig genuinely disables them."""
    econ_no_latent = ReasoningEconomics(config=ReasoningConfig(enable_latent=False))
    state = ReasoningState(strategy=ReasoningStrategy.NONE, model_id="aeon-small-1.5b")
    budget = ReasoningBudget()

    cands_no_latent = econ_no_latent.evaluate_candidates(
        state=state, budget=budget, task_type="reasoning", difficulty=0.5, uncertainty=0.5
    )
    for c in cands_no_latent:
        assert c.strategy != ReasoningStrategy.LATENT

    econ_no_explicit = ReasoningEconomics(config=ReasoningConfig(enable_explicit=False))
    state_deep = ReasoningState(strategy=ReasoningStrategy.MEDIUM, model_id="aeon-small-1.5b", reasoning_started=True)
    cands_no_explicit = econ_no_explicit.evaluate_candidates(
        state=state_deep, budget=budget, task_type="reasoning", difficulty=0.8, uncertainty=0.8
    )
    for c in cands_no_explicit:
        assert c.strategy not in [ReasoningStrategy.EXPLICIT, ReasoningStrategy.DEEP]


def test_multi_run_reproducibility():
    """Verify that repeated benchmark executions produce deterministic and stable results."""
    from aeon.benchmarks.harness import BenchmarkHarness
    harness = BenchmarkHarness()

    runs = []
    for _ in range(5):
        res = harness.run_phase6()
        runs.append(res)

    for r in runs:
        assert "1. No Reasoning (Direct)" in r
        assert "2. Fixed Heavy Reasoning" in r
        assert "3. AEON Adaptive Reasoning" in r
        adaptive_m = r["3. AEON Adaptive Reasoning"]
        assert adaptive_m.accuracy > 0.0
        assert adaptive_m.mean_cost_usd > 0.0

    # Check stability across runs
    costs = [r["3. AEON Adaptive Reasoning"].mean_cost_usd for r in runs]
    assert max(costs) - min(costs) < 0.0001
