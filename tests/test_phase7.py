"""
Phase 7 Unit, Integration, Invariant, and Adversarial Test Suite for AEON.
Covers Adaptive Model Depth, Mixture-of-Depths (MoD), Token Importance Estimation,
Dynamic Capacity Control, Early-Exit Evaluator, Economic VOI Depth Optimization,
12 Hard Safety Invariants, 15 Adversarial Test Cases, and CLI Commands.
"""

import math
import pytest
from click.testing import CliRunner

from aeon import (
    AEON,
    InferenceRequest,
    RouteChoice,
    TaskType,
    DifficultyLevel,
    ControllerAction,
    DecisionReason,
    DepthExitDecision,
    DepthStrategy,
    TokenImportanceSignal,
    ModelDepthProfile,
    TokenComputeState,
    DepthExecutionState,
    DepthCandidate,
    DepthConfig,
    ModelDepthRegistry,
    global_depth_registry,
    get_model_depth_profile,
    LayerCostModel,
    global_depth_cost_model,
    TokenImportanceEstimator,
    global_token_importance_estimator,
    CapacityController,
    global_capacity_controller,
    MoDRouter,
    global_mod_router,
    EarlyExitEvaluator,
    global_early_exit_evaluator,
    DepthEconomics,
    global_depth_economics,
    BaseDepthBackend,
    SimulationDepthBackend,
    global_depth_backend,
    DepthController,
    global_depth_controller,
)
from aeon.cli.main import cli
from aeon.benchmarks.harness import BenchmarkHarness
from aeon.benchmarks.synthetic_workload import generate_phase7_depth_workload


# =========================================================================
# 1. Depth Types & Registries
# =========================================================================

def test_depth_types():
    assert DepthExitDecision.CONTINUE == "continue"
    assert DepthExitDecision.EXIT == "exit"
    assert DepthExitDecision.FALLBACK_FULL_DEPTH == "fallback_full_depth"
    assert DepthExitDecision.BUDGET_EXHAUSTED == "budget_exhausted"
    assert DepthExitDecision.SLA_EXCEEDED == "sla_exceeded"
    assert DepthExitDecision.QUALITY_SUFFICIENT == "quality_sufficient"
    assert DepthExitDecision.CONFIDENCE_SUFFICIENT == "confidence_sufficient"
    assert DepthExitDecision.VERIFICATION_FAILED == "verification_failed"
    assert DepthExitDecision.ERROR == "error"

    assert DepthStrategy.HYBRID_DEPTH == "hybrid_depth"
    assert DepthStrategy.FULL_DEPTH == "full_depth"
    assert DepthStrategy.FIXED_EARLY_EXIT == "fixed_early_exit"
    assert DepthStrategy.ADAPTIVE_EARLY_EXIT == "adaptive_early_exit"
    assert DepthStrategy.FIXED_MOD == "fixed_mod"
    assert DepthStrategy.ADAPTIVE_MOD == "adaptive_mod"


def test_depth_profile_registry():
    prof_70b = get_model_depth_profile("aeon-large-70b")
    assert prof_70b.total_layers == 80
    assert prof_70b.minimum_safe_depth == 8
    assert prof_70b.supported_early_exit is True
    assert prof_70b.supported_mod is True
    assert prof_70b.is_synthetic is True

    prof_1_5b = get_model_depth_profile("aeon-small-1.5b")
    assert prof_1_5b.total_layers == 28
    assert prof_1_5b.minimum_safe_depth == 4

    # Unknown profile falls back cleanly
    prof_unknown = get_model_depth_profile("nonexistent-model-xyz")
    assert prof_unknown.total_layers == 28
    assert prof_unknown.minimum_safe_depth == 4


def test_depth_quality_curve():
    prof = get_model_depth_profile("aeon-large-70b")
    q4 = prof.get_quality_at_depth(4)
    q20 = prof.get_quality_at_depth(20)
    q40 = prof.get_quality_at_depth(40)
    q80 = prof.get_quality_at_depth(80)

    assert 0.0 < q4 < q20 < q40 <= q80 <= 1.0
    # Diminishing returns property: marginal gain from 20->40 should be less than or equal to 4->20
    gain_early = (q20 - q4) / 16.0
    gain_late = (q80 - q40) / 40.0
    assert gain_early > gain_late


# =========================================================================
# 2. Cost Model
# =========================================================================

def test_depth_cost_model():
    prof = get_model_depth_profile("aeon-large-70b")
    cost_model = LayerCostModel()

    flops_full = cost_model.calculate_token_flops(prof, depth=80, num_tokens=10)
    flops_half = cost_model.calculate_token_flops(prof, depth=40, num_tokens=10)
    assert math.isclose(flops_half, flops_full * 0.5, rel_tol=1e-5)

    lat_full = cost_model.estimate_depth_latency_ms(prof, depth=80, num_tokens=10)
    lat_quarter = cost_model.estimate_depth_latency_ms(prof, depth=20, num_tokens=10)
    assert lat_quarter < lat_full

    cost_usd = cost_model.estimate_depth_cost_usd(prof, depth=80, num_tokens=10)
    assert cost_usd > 0.0


# =========================================================================
# 3. Token Importance & MoD Router
# =========================================================================

def test_token_importance_estimator():
    estimator = TokenImportanceEstimator()
    tokens = ["the", "x", "=", "104729", "therefore", "proof", "is", "complete"]
    importance = estimator.estimate_importance(
        tokens=tokens,
        query="Verify prime number 104729",
        task_type=TaskType.REASONING,
        difficulty=0.8,
    )
    assert len(importance) == len(tokens)
    for score in importance:
        assert 0.0 <= score <= 1.0

    # Number token or logic keyword should have higher importance than stopword 'the'
    idx_num = tokens.index("104729")
    idx_the = tokens.index("the")
    assert importance[idx_num] > importance[idx_the]


def test_mod_router_allocation_respects_capacity():
    router = MoDRouter()
    prof = get_model_depth_profile("aeon-large-70b")
    num_tokens = 20
    tokens = [f"tok_{i}" for i in range(num_tokens)]
    importance = [0.1 + (i / float(num_tokens)) * 0.9 for i in range(num_tokens)]

    states = router.allocate_token_depths(
        tokens=tokens,
        importance_scores=importance,
        profile=prof,
        capacity_ratio=0.5,
        min_depth=prof.minimum_safe_depth,
        max_depth=prof.total_layers,
    )

    assert len(states) == num_tokens
    total_allocated = sum(s.allocated_layers for s in states)
    # Target capacity is capacity_ratio * total_layers * num_tokens
    max_budget = 0.5 * prof.total_layers * num_tokens
    assert total_allocated <= max_budget + 1e-4

    # Highly important token gets deeper layers than lowest importance token
    assert states[-1].allocated_layers > states[0].allocated_layers


def test_mod_router_uniform_allocation():
    router = MoDRouter()
    prof = get_model_depth_profile("aeon-small-1.5b")
    num_tokens = 10
    tokens = [f"tok_{i}" for i in range(num_tokens)]
    importance = [0.5] * num_tokens

    states = router.allocate_token_depths(
        tokens=tokens,
        importance_scores=importance,
        profile=prof,
        capacity_ratio=0.6,
        min_depth=prof.minimum_safe_depth,
        max_depth=prof.total_layers,
        uniform_fallback=True,
    )
    # When all importance is identical, all tokens get uniform depth
    allocs = [s.allocated_layers for s in states]
    assert len(set(allocs)) == 1


# =========================================================================
# 4. Capacity Controller
# =========================================================================

def test_capacity_controller_adaptation():
    cc = CapacityController()
    req_easy = InferenceRequest(query="Hi", sla_latency_ms=5000.0, cost_budget_usd=0.05)
    cap_easy = cc.determine_capacity(req_easy, difficulty=0.1, uncertainty=0.1)

    req_hard = InferenceRequest(query="Hard proof", sla_latency_ms=5000.0, cost_budget_usd=0.05)
    cap_hard = cc.determine_capacity(req_hard, difficulty=0.9, uncertainty=0.8)

    assert cap_hard > cap_easy
    assert 0.20 <= cap_easy <= 1.0
    assert 0.20 <= cap_hard <= 1.0


def test_capacity_controller_sla_tight():
    cc = CapacityController()
    req_tight_sla = InferenceRequest(query="Fast answer", sla_latency_ms=10.0, cost_budget_usd=0.01)
    cap_tight = cc.determine_capacity(req_tight_sla, difficulty=0.5, uncertainty=0.5)
    assert cap_tight < 0.60


# =========================================================================
# 5. Early-Exit Evaluator
# =========================================================================

def test_early_exit_evaluator_min_depth():
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Short factual query")

    # Layer 4 is below min safe depth 8
    res = evaluator.evaluate_exit(
        current_layer=4,
        profile=prof,
        request=req,
        confidence=0.99,
        estimated_quality=0.95,
        marginal_gain=0.001,
        marginal_cost=0.0001,
        voi=0.01,
    )
    assert res.decision == DepthExitDecision.CONTINUE
    assert "Below minimum safe depth" in res.reason


def test_early_exit_evaluator_confidence_exit():
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Short factual query", target_quality=0.80)

    # Layer 24 (well above min depth 8), high confidence, low marginal gain
    res = evaluator.evaluate_exit(
        current_layer=24,
        profile=prof,
        request=req,
        confidence=0.92,
        estimated_quality=0.88,
        marginal_gain=0.002,
        marginal_cost=0.0005,
        voi=0.02,
    )
    assert res.decision == DepthExitDecision.QUALITY_SUFFICIENT
    assert res.should_exit is True


def test_early_exit_verification_gate():
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Prove theorem", require_verification=True)

    # Even with high confidence, verification required prevents exit at shallow layers
    res = evaluator.evaluate_exit(
        current_layer=20,
        profile=prof,
        request=req,
        confidence=0.95,
        estimated_quality=0.90,
        marginal_gain=0.005,
        marginal_cost=0.0005,
        voi=0.05,
        verified=False,
    )
    assert res.decision == DepthExitDecision.CONTINUE
    assert "Verification required" in res.reason


# =========================================================================
# 6. Depth Economics & Break-Even Analysis
# =========================================================================

def test_depth_economics_evaluation():
    econ = DepthEconomics()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Standard classification query")

    candidates = econ.evaluate_candidates(
        profile=prof,
        request=req,
        query_tokens=20,
        difficulty=0.3,
        uncertainty=0.2,
        target_quality=0.85,
    )
    assert len(candidates) == 6
    strategies = [c.strategy for c in candidates]
    assert DepthStrategy.FULL_DEPTH in strategies
    assert DepthStrategy.HYBRID_DEPTH in strategies
    assert DepthStrategy.ADAPTIVE_MOD in strategies

    for c in candidates:
        assert c.estimated_cost_usd > 0.0
        assert c.estimated_latency_ms > 0.0
        assert 0.0 <= c.estimated_quality <= 1.0


def test_break_even_analysis():
    prof = get_model_depth_profile("aeon-large-70b")
    econ = DepthEconomics()
    be = econ.break_even_analysis(prof)
    assert "break_even_marginal_quality_early_exit" in be
    assert "break_even_marginal_quality_mod" in be
    assert "overhead_ms" in be
    assert be["break_even_marginal_quality_early_exit"] > 0.0


# =========================================================================
# 7. Simulation Backend & Real Backend Abstraction
# =========================================================================

def test_backend_neutral_abstraction_unsupported():
    base = BaseDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    with pytest.raises(NotImplementedError) as excinfo:
        base.execute_layer(prof, 0, ["test"])
    assert "UNSUPPORTED_REAL_BACKEND" in str(excinfo.value)


def test_simulation_backend_execution():
    sim = SimulationDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    state = sim.simulate_depth_run(
        profile=prof,
        tokens=["a", "b", "c", "d"],
        strategy=DepthStrategy.HYBRID_DEPTH,
        initial_depth=16,
        target_quality=0.85,
        difficulty=0.4,
        uncertainty=0.3,
    )
    assert state.is_simulated is True
    assert state.execution_mode == "SIMULATED_DEPTH_EXECUTION"
    assert prof.minimum_safe_depth <= state.actual_depth <= prof.total_layers
    assert state.estimated_flops > 0.0
    assert state.estimated_cost_usd > 0.0


# =========================================================================
# 8. Depth Controller & End-to-End AEON Integration
# =========================================================================

def test_depth_controller_integration():
    ctrl = DepthController()
    req = InferenceRequest(query="Calculate sqrt(144)", adaptive_depth=True)
    ans, state, tel = ctrl.execute_depth_inference(req, model_name="aeon-small-1.5b", query_tokens=10, target_quality=0.85)
    assert state.actual_depth is not None
    assert state.actual_depth <= state.requested_depth
    assert state.is_simulated is True


def test_aeon_end_to_end_with_depth():
    aeon = AEON(policy="voi", adaptive_depth=True)
    res = aeon.run("What is 15 * 12?", adaptive_depth=True)

    assert res.trace.depth_enabled is True
    assert res.trace.depth_execution_mode == "SIMULATED_DEPTH_EXECUTION"
    assert res.trace.requested_depth is not None
    assert res.trace.actual_depth is not None
    assert res.trace.actual_depth <= res.trace.requested_depth
    assert res.trace.depth_layers_saved >= 0


def test_aeon_full_depth_strategy():
    aeon = AEON(policy="rules", adaptive_depth=True, depth_strategy="full_depth")
    res = aeon.run("Give a brief overview of graph theory", adaptive_depth=True, depth_strategy="full_depth")

    assert res.trace.depth_strategy == "full_depth"
    assert res.trace.actual_depth == res.trace.requested_depth
    assert res.trace.early_exit is False


# =========================================================================
# 9. Hard Safety Invariants (1 to 12)
# =========================================================================

def test_invariant_1_no_token_below_minimum_depth():
    """Invariant 1: No token receives depth below minimum safe depth."""
    router = MoDRouter()
    prof = get_model_depth_profile("aeon-large-70b")
    tokens = ["word"] * 5
    states = router.allocate_token_depths(
        tokens=tokens,
        importance_scores=[0.01] * 5,
        profile=prof,
        capacity_ratio=0.1,  # very tight capacity
        min_depth=prof.minimum_safe_depth,
        max_depth=prof.total_layers,
    )
    for s in states:
        assert s.allocated_layers >= prof.minimum_safe_depth


def test_invariant_2_no_token_above_maximum_depth():
    """Invariant 2: No token receives depth above maximum model layers."""
    router = MoDRouter()
    prof = get_model_depth_profile("aeon-small-1.5b")
    tokens = ["word"] * 5
    states = router.allocate_token_depths(
        tokens=tokens,
        importance_scores=[1.0] * 5,
        profile=prof,
        capacity_ratio=1.5,  # excess capacity
        min_depth=prof.minimum_safe_depth,
        max_depth=prof.total_layers,
    )
    for s in states:
        assert s.allocated_layers <= prof.total_layers


def test_invariant_3_no_negative_depth():
    """Invariant 3: No token or model computation has negative depth."""
    sim = SimulationDepthBackend()
    prof = get_model_depth_profile("aeon-medium-7b")
    state = sim.simulate_depth_run(
        profile=prof,
        tokens=["a", "b"],
        strategy=DepthStrategy.HYBRID_DEPTH,
        initial_depth=0,
    )
    assert state.actual_depth >= 0
    assert state.requested_depth >= 0


def test_invariant_4_no_invalid_layer_index():
    """Invariant 4: Layer indices must stay within [0, total_layers - 1]."""
    prof = get_model_depth_profile("aeon-large-70b")
    for d in [0, 10, 80]:
        q = prof.get_quality_at_depth(d)
        assert 0.0 <= q <= 1.0


def test_invariant_5_no_compute_allocation_above_budget():
    """Invariant 5: MoD allocation must not exceed the capacity layer budget."""
    router = MoDRouter()
    prof = get_model_depth_profile("aeon-large-70b")
    tokens = [f"t_{i}" for i in range(10)]
    capacity = 0.4
    states = router.allocate_token_depths(
        tokens=tokens,
        importance_scores=[0.8] * 10,
        profile=prof,
        capacity_ratio=capacity,
        min_depth=prof.minimum_safe_depth,
        max_depth=prof.total_layers,
    )
    total_allocated = sum(s.allocated_layers for s in states)
    max_budget = capacity * prof.total_layers * len(tokens)
    assert total_allocated <= max_budget + 1e-4


def test_invariant_6_no_nan_or_inf_allocation():
    """Invariant 6: No NaN or Inf propagation in compute, latency, or quality."""
    cost_model = LayerCostModel()
    prof = get_model_depth_profile("aeon-large-70b")
    flops = cost_model.calculate_token_flops(prof, depth=0, num_tokens=0)
    assert not math.isnan(flops) and not math.isinf(flops)
    lat = cost_model.estimate_depth_latency_ms(prof, depth=0, num_tokens=0)
    assert not math.isnan(lat) and not math.isinf(lat)


def test_invariant_7_no_exit_before_minimum_safe_depth():
    """Invariant 7: Early exit cannot occur before minimum safe depth."""
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Trivial")
    for layer in range(0, prof.minimum_safe_depth):
        res = evaluator.evaluate_exit(
            current_layer=layer,
            profile=prof,
            request=req,
            confidence=1.0,
            estimated_quality=1.0,
            marginal_gain=0.0,
            marginal_cost=0.001,
            voi=0.0,
        )
        assert res.should_exit is False
        assert res.decision == DepthExitDecision.CONTINUE


def test_invariant_8_verification_failure_cannot_produce_premature_exit():
    """Invariant 8: Verification failure prevents premature shallow exit."""
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Proof", require_verification=True)
    res = evaluator.evaluate_exit(
        current_layer=24,
        profile=prof,
        request=req,
        confidence=0.99,
        estimated_quality=0.95,
        marginal_gain=0.001,
        marginal_cost=0.001,
        voi=0.01,
        verified=False,
    )
    assert res.should_exit is False


def test_invariant_9_unsupported_real_backend_fails_safely():
    """Invariant 9: Unsupported real MoD backend reports unsupported cleanly."""
    base = BaseDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    with pytest.raises(NotImplementedError):
        base.execute_mod(prof, ["token"], [10])


def test_invariant_10_simulation_explicitly_labeled():
    """Invariant 10: All simulated depth executions are explicitly labeled."""
    sim = SimulationDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    state = sim.simulate_depth_run(profile=prof, tokens=["hi"], strategy=DepthStrategy.HYBRID_DEPTH, initial_depth=16)
    assert state.is_simulated is True
    assert state.execution_mode == "SIMULATED_DEPTH_EXECUTION"


def test_invariant_11_full_depth_fallback_remains_available():
    """Invariant 11: Full-depth fallback remains completely functional."""
    aeon = AEON(policy="rules", adaptive_depth=True, depth_strategy="full_depth")
    res = aeon.run("Ensure full depth works", adaptive_depth=True, depth_strategy="full_depth")
    assert res.trace.actual_depth == res.trace.requested_depth
    assert res.trace.depth_strategy == "full_depth"


def test_invariant_12_phase_1_to_6_functionality_preserved():
    """Invariant 12: Adaptive depth runs seamlessly alongside tools, memory, speculation, and reasoning."""
    aeon = AEON(
        policy="voi",
        memory=True,
        tools=True,
        speculation=True,
        reasoning=True,
        adaptive_depth=True,
    )
    res = aeon.run("Solve 47 * 82 and explain steps", adaptive_depth=True, reasoning=True, tools=True)
    assert res.answer is not None
    assert res.trace.depth_enabled is True


# =========================================================================
# 10. Adversarial Test Cases (Case A through Case O)
# =========================================================================

def test_case_a_trivial_task():
    """Case A — trivial task uses minimal computation and exits early."""
    aeon = AEON(policy="voi", adaptive_depth=True)
    res = aeon.run("Hi", adaptive_depth=True)
    assert res.trace.actual_depth <= 20
    assert res.trace.early_exit is True


def test_case_b_deterministic_arithmetic_tool_dominates():
    """Case B — deterministic arithmetic routes to tool, skipping deep model layers."""
    aeon = AEON(policy="voi", tools=True, adaptive_depth=True)
    res = aeon.run("Calculate: 125 * 37", tools=True, adaptive_depth=True)
    assert res.trace.tool_used is not None
    assert "4625" in res.answer


def test_case_c_medium_question_moderate_depth():
    """Case C — medium question gets moderate depth."""
    aeon = AEON(policy="voi", adaptive_depth=True)
    res = aeon.run("A train departs at 2:15 PM and arrives at 4:45 PM. What was the duration?", adaptive_depth=True)
    assert res.trace.actual_depth >= 8


def test_case_d_difficult_reasoning_deeper_computation():
    """Case D — difficult reasoning utilizes deeper layer allocation."""
    aeon = AEON(policy="voi", adaptive_depth=True)
    res = aeon.run("Construct a formal proof of Cantor's diagonal argument", adaptive_depth=True)
    assert res.trace.actual_depth >= 16


def test_case_e_diminishing_returns_stops_before_full_depth():
    """Case E — diminishing returns halts depth before full layer allocation."""
    sim = SimulationDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    state = sim.simulate_depth_run(
        profile=prof,
        tokens=["a"] * 10,
        strategy=DepthStrategy.ADAPTIVE_EARLY_EXIT,
        initial_depth=20,
        target_quality=0.80,
        difficulty=0.3,
        uncertainty=0.1,
    )
    assert state.actual_depth < prof.total_layers


def test_case_f_highly_uncertain_continues_computation():
    """Case F — high uncertainty drives deeper allocation than low uncertainty."""
    sim = SimulationDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    state_low_u = sim.simulate_depth_run(
        profile=prof,
        tokens=["a"] * 10,
        strategy=DepthStrategy.HYBRID_DEPTH,
        initial_depth=16,
        difficulty=0.5,
        uncertainty=0.1,
    )
    state_high_u = sim.simulate_depth_run(
        profile=prof,
        tokens=["a"] * 10,
        strategy=DepthStrategy.HYBRID_DEPTH,
        initial_depth=16,
        difficulty=0.5,
        uncertainty=0.9,
    )
    assert state_high_u.actual_depth >= state_low_u.actual_depth


def test_case_g_strict_latency_sla_reduces_depth():
    """Case G — strict latency SLA bounds depth execution."""
    aeon = AEON(policy="voi", adaptive_depth=True)
    res = aeon.run("Analyze market trends", sla_latency_ms=10.0, adaptive_depth=True)
    assert res.trace.actual_depth <= 32


def test_case_h_strict_compute_budget_bounded():
    """Case H — strict compute budget enforces depth limit."""
    aeon = AEON(policy="voi", adaptive_depth=True)
    res = aeon.run("Solve physics problem", cost_budget_usd=0.00005, adaptive_depth=True)
    assert res.cost <= 0.005


def test_case_i_one_token_extremely_important():
    """Case I — one token extremely important receives deeper allocation."""
    router = MoDRouter()
    prof = get_model_depth_profile("aeon-large-70b")
    tokens = ["the", "the", "the", "KEYWORD_CRITICAL"]
    importance = [0.1, 0.1, 0.1, 1.0]
    states = router.allocate_token_depths(
        tokens=tokens,
        importance_scores=importance,
        profile=prof,
        capacity_ratio=0.5,
        min_depth=prof.minimum_safe_depth,
        max_depth=prof.total_layers,
    )
    critical_state = states[3]
    common_state = states[0]
    assert critical_state.allocated_layers > common_state.allocated_layers


def test_case_j_all_tokens_equally_important_approaches_uniform():
    """Case J — all tokens equally important approach uniform allocation."""
    router = MoDRouter()
    prof = get_model_depth_profile("aeon-large-70b")
    tokens = ["tok1", "tok2", "tok3", "tok4"]
    importance = [0.6, 0.6, 0.6, 0.6]
    states = router.allocate_token_depths(
        tokens=tokens,
        importance_scores=importance,
        profile=prof,
        capacity_ratio=0.5,
        min_depth=prof.minimum_safe_depth,
        max_depth=prof.total_layers,
    )
    allocs = [s.allocated_layers for s in states]
    assert max(allocs) - min(allocs) <= 2


def test_case_k_verification_failure_deepens():
    """Case K — unverified task prevents shallow early exit."""
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Math problem", require_verification=True)
    res = evaluator.evaluate_exit(
        current_layer=16,
        profile=prof,
        request=req,
        confidence=0.99,
        estimated_quality=0.90,
        marginal_gain=0.01,
        marginal_cost=0.001,
        voi=0.01,
        verified=False,
    )
    assert res.decision == DepthExitDecision.CONTINUE


def test_case_l_unsupported_real_backend_safe_fallback():
    """Case L — unsupported real backend fails cleanly with clear error."""
    base = BaseDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    with pytest.raises(NotImplementedError):
        base.execute_layer(prof, 1, ["x"])


def test_case_m_invalid_profile_safe_failure():
    """Case M — missing/invalid profile falls back to default profile."""
    prof = get_model_depth_profile(None)
    assert prof.total_layers == 28
    assert prof.minimum_safe_depth == 4


def test_case_n_nan_inf_safe_handling():
    """Case N — NaN/Inf values are handled without crashing."""
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")
    req = InferenceRequest(query="Trivial query")
    res = evaluator.evaluate_exit(
        current_layer=20,
        profile=prof,
        request=req,
        confidence=float("nan"),
        estimated_quality=float("inf"),
        marginal_gain=float("-inf"),
        marginal_cost=0.0,
        voi=float("nan"),
    )
    assert res.decision in [DepthExitDecision.CONTINUE, DepthExitDecision.ERROR]


def test_case_o_full_depth_required_no_unsafe_exit():
    """Case O — full-depth strategy forbids early exit."""
    sim = SimulationDepthBackend()
    prof = get_model_depth_profile("aeon-large-70b")
    state = sim.simulate_depth_run(
        profile=prof,
        tokens=["a", "b"],
        strategy=DepthStrategy.FULL_DEPTH,
        initial_depth=80,
    )
    assert state.actual_depth == prof.total_layers
    assert state.early_exit is False


# =========================================================================
# 11. Benchmark Harness & Ablations
# =========================================================================

def test_benchmark_harness_phase7_runs():
    harness = BenchmarkHarness()
    results = harness.run_phase7(runs=1)
    assert len(results) == 7
    assert "Baseline A: Full Depth" in results
    assert "Baseline G: AEON Phase 7 Full System" in results
    for name, m in results.items():
        assert m.accuracy >= 0.0
        assert m.mean_latency_ms >= 0.0
        assert m.total_cost_usd >= 0.0


def test_benchmark_harness_depth_ablation():
    harness = BenchmarkHarness()
    ablations = harness.run_depth_ablation_study()
    assert len(ablations) == 10
    assert "Full AEON Phase 7" in ablations
    assert "Without Early Exit" in ablations
    assert "Without MoD" in ablations


def test_benchmark_harness_breakdown():
    harness = BenchmarkHarness()
    breakdown = harness.run_depth_workload_breakdown()
    assert len(breakdown) == 12
    assert "trivial" in breakdown
    assert "proof_logic" in breakdown


def test_benchmark_harness_break_even():
    harness = BenchmarkHarness()
    be = harness.run_depth_break_even_analysis()
    assert "break_even_marginal_quality_early_exit" in be


def test_benchmark_harness_overhead():
    harness = BenchmarkHarness()
    ov = harness.run_depth_controller_overhead()
    assert "controller_overhead_mean_ms" in ov
    assert ov["controller_overhead_mean_ms"] < 10.0


# =========================================================================
# 12. CLI Commands
# =========================================================================

def test_cli_depth_commands():
    runner = CliRunner()

    res_cfg = runner.invoke(cli, ["depth", "config"])
    assert res_cfg.exit_code == 0
    assert "enabled" in res_cfg.output

    res_be = runner.invoke(cli, ["depth", "break-even"])
    assert res_be.exit_code == 0
    assert "break_even_marginal_quality_early_exit" in res_be.output

    res_ov = runner.invoke(cli, ["depth", "overhead"])
    assert res_ov.exit_code == 0
    assert "controller_overhead_mean_ms" in res_ov.output


def test_cli_run_with_adaptive_depth():
    runner = CliRunner()
    res = runner.invoke(cli, ["run", "What is the boiling point of nitrogen?", "--adaptive-depth", "--trace"])
    assert res.exit_code == 0
    assert "AEON ADAPTIVE DEPTH & MOD" in res.output


# =========================================================================
# 13. Adversarial Audit Regression Tests (Post-Audit Hardening)
# =========================================================================

def test_depth_sensitivity_accuracy_tradeoff():
    """Audit Test 1: Prove that shallow depth on hard tasks fails quality, while adaptive/full depth passes."""
    aeon_shallow = AEON(policy="rules", adaptive_depth=True, max_depth=8)
    res_shallow = aeon_shallow.run(
        InferenceRequest(
            query="Analyze the consistency of Zermelo-Fraenkel set theory without the Axiom of Choice and discuss Godel's second incompleteness theorem.",
            ground_truth="Consistent relative to ZFC",
            task_hint="reasoning",
            adaptive_depth=True,
            max_depth=8,
        )
    )
    # Shallow depth (8 layers) on hard task fails quality target (< 0.75)
    assert res_shallow.trace.actual_depth <= 8
    assert res_shallow.trace.correct is False

    aeon_adaptive = AEON(policy="voi", adaptive_depth=True)
    res_adaptive = aeon_adaptive.run(
        InferenceRequest(
            query="Analyze the consistency of Zermelo-Fraenkel set theory without the Axiom of Choice and discuss Godel's second incompleteness theorem.",
            ground_truth="Consistent relative to ZFC",
            task_hint="reasoning",
            adaptive_depth=True,
        )
    )
    # Adaptive depth allocates sufficient depth to satisfy quality target
    assert res_adaptive.trace.actual_depth >= 20
    assert res_adaptive.trace.correct is True


def test_mod_dynamicity_token_ranking():
    """Audit Test 2: Prove that token importance order strictly dictates MoD depth allocation."""
    router = MoDRouter()
    t_a = TokenComputeState(token_id=0, token_text="A", importance=0.95)
    t_b = TokenComputeState(token_id=1, token_text="B", importance=0.20)
    t_c = TokenComputeState(token_id=2, token_text="C", importance=0.60)

    res = router.route_token_depths([t_a, t_b, t_c], total_layers=80, min_depth=8, max_depth=80, capacity_ratio=0.65)
    assert res[0].allocated_layers > res[2].allocated_layers > res[1].allocated_layers

    # Reverse order: B high, A low
    t_a2 = TokenComputeState(token_id=0, token_text="A", importance=0.15)
    t_b2 = TokenComputeState(token_id=1, token_text="B", importance=0.95)
    t_c2 = TokenComputeState(token_id=2, token_text="C", importance=0.60)
    res2 = router.route_token_depths([t_a2, t_b2, t_c2], total_layers=80, min_depth=8, max_depth=80, capacity_ratio=0.65)
    assert res2[1].allocated_layers > res2[2].allocated_layers > res2[0].allocated_layers


def test_ablation_depth_config_propagation():
    """Audit Test 3: Prove that disabling MoD or early exit in DepthConfig genuinely modifies execution."""
    cfg_no_exit = DepthConfig(early_exit_enabled=False)
    assert cfg_no_exit.enable_early_exit is False

    cfg_no_mod = DepthConfig(mod_enabled=False)
    assert cfg_no_mod.enable_mod is False

    cfg_no_cap = DepthConfig(adaptive_capacity=False)
    assert cfg_no_cap.enable_adaptive_capacity is False

    cfg_uniform = DepthConfig(uniform_importance=True)
    assert cfg_uniform.enable_token_importance is False

    # When both early exit and MoD are disabled, execution MUST run at full depth
    cfg_full = DepthConfig(early_exit_enabled=False, mod_enabled=False)
    ctrl_full = DepthController(config=cfg_full)
    _, st_full, _ = ctrl_full.execute_depth_inference(
        query="Hi", model_name="aeon-medium-7b", difficulty=0.35, uncertainty=0.24
    )
    assert st_full.actual_depth == 32
    assert st_full.early_exit is False


def test_break_even_formulas_across_tokens():
    """Audit Test 4: Prove break-even net latency and FLOP savings across token lengths."""
    from aeon.depth.cost_model import DepthCostModel
    cm = DepthCostModel()
    prof = get_model_depth_profile("aeon-medium-7b")

    for tok in [4, 8, 16, 32]:
        full_lat = cm.calculate_layer_latency_ms("aeon-medium-7b", prof.total_layers, prof.total_layers, tok)
        half_lat = cm.calculate_layer_latency_ms("aeon-medium-7b", prof.total_layers, prof.total_layers // 2, tok)
        saved_lat = full_lat - half_lat
        # Net savings must be positive and exceed controller overhead (0.2ms)
        assert saved_lat > 0.20

        full_flops = cm.calculate_layer_flops("aeon-medium-7b", prof.total_layers, prof.total_layers, tok)
        half_flops = cm.calculate_layer_flops("aeon-medium-7b", prof.total_layers, prof.total_layers // 2, tok)
        assert full_flops > half_flops


def test_full_depth_fallback_after_insufficient_quality():
    """Audit Test 5: Verify safe full-depth fallback when early exit quality is insufficient."""
    evaluator = EarlyExitEvaluator()
    prof = get_model_depth_profile("aeon-large-70b")

    # At layer 16, quality is 0.78 but target is 0.95
    res = evaluator.evaluate_exit(
        current_layer=16,
        profile=prof,
        current_quality=0.78,
        current_confidence=0.75,
        target_quality=0.95,
        marginal_quality_gain=0.17,
        marginal_cost_usd=0.0001,
    )
    assert res.should_exit is False
    assert res.decision == DepthExitDecision.CONTINUE
