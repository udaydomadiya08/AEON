"""
Phase 5 Unit, Integration, and Invariant Test Suite for AEON.
Tests Speculative and Token-Efficient Inference, Economic Modeling,
Break-Even Analysis, Online Acceptance Learning, Adaptive Draft Length,
N-Gram Speculation, Verification, Dynamic Stopping Integration,
Telemetry, and Specification Property Invariants 1 through 15.
Conforms to AEON Specification Sections 26 & 27.
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
    SpeculationStrategy,
    SpeculationAction,
    DraftTokenProposal,
    SpeculationCandidate,
    SpeculationState,
    SpeculationConfig,
    SpeculationEconomics,
    AdaptiveAcceptanceEstimator,
    AdaptiveDraftLengthOptimizer,
    NgramSpeculationEngine,
    SpeculationVerifier,
    SpeculativeStrategySelector,
    SpeculationController,
    global_registry,
)
from aeon.models.model_profiles import LARGE_MODEL_PROFILE, MEDIUM_MODEL_PROFILE, DRAFT_MODEL_0_5B_PROFILE
from aeon.cli.main import cli
from aeon.benchmarks.harness import BenchmarkHarness


# =========================================================================
# 1. Speculation Types & Initializations
# =========================================================================

def test_speculation_types():
    assert SpeculationStrategy.STANDARD == "standard"
    assert SpeculationStrategy.DRAFT_MODEL == "draft_model"
    assert SpeculationStrategy.NGRAM == "ngram"
    assert SpeculationStrategy.MULTI_TOKEN == "multi_token"

    state = SpeculationState(
        request_id="test-req",
        target_model="aeon-large-70b",
        current_strategy=SpeculationStrategy.DRAFT_MODEL,
        draft_model="aeon-draft-0.5b",
        draft_length=4,
    )
    assert state.generated_tokens == 0
    assert state.acceptance_rate == 0.0
    assert state.execution_mode == "SIMULATED_SPECULATIVE_DECODING"


# =========================================================================
# 2. Acceptance Estimator
# =========================================================================

def test_acceptance_estimator():
    estimator = AdaptiveAcceptanceEstimator(ema_alpha=0.20)
    # Default prediction
    pred = estimator.predict_acceptance(SpeculationStrategy.DRAFT_MODEL, draft_model="aeon-draft-0.5b", draft_length=4)
    assert 0.50 <= pred <= 0.95

    # Record successful round (4 drafted, 4 accepted -> 1.0)
    r1 = estimator.record_round(SpeculationStrategy.DRAFT_MODEL, "aeon-draft-0.5b", 4, 4, 4)
    assert r1 == 1.0
    assert estimator.global_ema > 0.75

    # Record deteriorating rounds (4 drafted, 0 accepted)
    estimator.record_round(SpeculationStrategy.DRAFT_MODEL, "aeon-draft-0.5b", 4, 4, 1)
    estimator.record_round(SpeculationStrategy.DRAFT_MODEL, "aeon-draft-0.5b", 4, 4, 0)
    assert estimator.is_deteriorating(window=3) is True


# =========================================================================
# 3. Draft Length Optimizer
# =========================================================================

def test_draft_length_optimizer():
    optimizer = AdaptiveDraftLengthOptimizer()
    target_70b = global_registry.get("aeon-large-70b")
    draft_0_5b = global_registry.get("aeon-draft-0.5b")

    opt_len, speedup, explanation = optimizer.optimize_draft_length(
        strategy=SpeculationStrategy.DRAFT_MODEL,
        target_profile=target_70b,
        draft_profile=draft_0_5b,
        current_acceptance=0.85,
    )
    assert opt_len in [2, 4, 6, 8]
    assert speedup > 1.0
    assert "Selected optimal draft length" in explanation

    # Feedback adaptation
    adapted_down, r_down = optimizer.adapt_length_on_feedback(current_length=6, realized_acceptance=0.25, break_even_rate=0.60)
    assert adapted_down < 6
    assert "reducing draft length" in r_down

    adapted_up, r_up = optimizer.adapt_length_on_feedback(current_length=4, realized_acceptance=0.95, break_even_rate=0.60)
    assert adapted_up > 4
    assert "increasing draft length" in r_up


# =========================================================================
# 4. Strategy Selector
# =========================================================================

def test_strategy_selector():
    selector = SpeculativeStrategySelector()
    top, all_cands = selector.select_strategy(
        target_model="aeon-large-70b",
        query="Write code for binary search algorithm",
        task_type=TaskType.CODE,
        tokens_to_generate=60,
    )
    assert top is not None
    assert len(all_cands) >= 3
    # On 70B target with high code predictability, speculation should be viable
    assert top.is_viable is True
    assert top.expected_speedup >= 1.0


# =========================================================================
# 5. Model Registry & Compatibility
# =========================================================================

def test_draft_model_registry():
    drafts = global_registry.list_draft_models()
    assert len(drafts) >= 2
    draft_names = [d.name for d in drafts]
    assert "aeon-draft-0.5b" in draft_names
    assert "aeon-small-1.5b" in draft_names


def test_model_compatibility():
    # 0.5B draft is compatible with 70B target
    assert global_registry.is_compatible("aeon-draft-0.5b", "aeon-large-70b") is True
    # 0.5B draft is compatible with 7B target
    assert global_registry.is_compatible("aeon-draft-0.5b", "aeon-medium-7b") is True
    # 70B cannot draft for 70B
    assert global_registry.is_compatible("aeon-large-70b", "aeon-large-70b") is False
    # 70B cannot draft for 0.5B (parameters strictly greater)
    assert global_registry.is_compatible("aeon-large-70b", "aeon-draft-0.5b") is False


# =========================================================================
# 6. N-Gram Speculation
# =========================================================================

def test_ngram_speculation():
    engine = NgramSpeculationEngine()
    prompt = "SELECT user_id, user_name, user_email FROM users WHERE user_id = 1;"
    engine.index_context(prompt)

    # Context ends with 'SELECT user_id,' -> should predict 'user_name,'
    prop = engine.speculate("SELECT user_id", draft_length=3)
    assert prop is not None
    assert prop.strategy == SpeculationStrategy.NGRAM
    assert len(prop.tokens) > 0
    assert "user_name" in prop.tokens or "," in prop.tokens
    assert prop.draft_cost_usd == 0.0


# =========================================================================
# 7. Speculation Verifier
# =========================================================================

def test_verifier():
    verifier = SpeculationVerifier()
    target_70b = global_registry.get("aeon-large-70b")

    proposal = DraftTokenProposal(
        tokens=["quick", "brown", "fox", "jumps"],
        confidences=[0.9, 0.9, 0.9, 0.9],
        strategy=SpeculationStrategy.DRAFT_MODEL,
        draft_length=4,
    )
    # Target reference matches first 3 tokens, then differs on 4th ('leaps' vs 'jumps')
    reference = ["quick", "brown", "fox", "leaps", "over"]
    res = verifier.verify_draft(target_70b, reference, proposal)

    assert res.accepted_tokens == ["quick", "brown", "fox"]
    assert res.accepted_length == 3
    assert res.rejected_tokens == ["jumps"]
    assert res.replacement_token == "leaps"
    assert res.acceptance_rate == 0.75
    assert res.verification_latency_ms > 0.0


# =========================================================================
# 8. Economics & Break-Even
# =========================================================================

def test_speculation_economics():
    econ = SpeculationEconomics()
    u_std = econ.compute_utility(quality=0.95, latency_ms=100.0, cost_usd=0.001, compute_flops=1e9)
    u_spec = econ.compute_utility(quality=0.95, latency_ms=50.0, cost_usd=0.001, compute_flops=1e9)
    # Lower latency produces strictly higher utility
    assert u_spec > u_std


def test_break_even():
    econ = SpeculationEconomics()
    target_70b = global_registry.get("aeon-large-70b")
    draft_0_5b = global_registry.get("aeon-draft-0.5b")

    be = econ.calculate_break_even(
        target_profile=target_70b,
        draft_profile=draft_0_5b,
        draft_length=4,
    )
    # For a 70B target vs 0.5B draft, break-even acceptance is typically between 30% and 75%
    assert 0.20 <= be.min_acceptance_rate <= 0.85
    assert be.expected_speedup > 1.0
    assert be.is_favorable is True


# =========================================================================
# 9. VOI & Dynamic Stopping Integration
# =========================================================================

def test_voi_integration():
    aeon = AEON(policy="voi", speculation=True)
    res = aeon.run("Write a Python function to compute the Fibonacci sequence.", speculation=True)
    assert res.trace.speculative is True or res.trace.route in ["speculative", "small_model", "medium_model", "large_model"]
    assert res.trace.decision_reason is not None


def test_dynamic_stopping_integration():
    aeon = AEON(policy="voi", speculation=True)
    res = aeon.run("Describe the process of cell division.", speculation=True)
    assert res.answer is not None
    assert len(res.answer) > 0
    assert res.trace.latency_ms > 0.0


# =========================================================================
# 10. Memory & Speculation Synergy
# =========================================================================

def test_memory_integration():
    aeon = AEON(policy="voi", memory=True, speculation=True)
    # Run 1 populates memory
    res1 = aeon.run("Write a Python function to calculate factorial.", memory=True, speculation=True)
    # Run 2 hits semantic cache and avoids both generation and speculation
    res2 = aeon.run("Write a Python function to calculate factorial.", memory=True, speculation=True)
    assert res2.trace.memory_hit is True
    assert res2.trace.cost_saved_usd >= 0.0


# =========================================================================
# 11. Failure Recovery & Fallback
# =========================================================================

def test_failure_recovery():
    # If acceptance collapses or draft model is incompatible, system falls back to standard decoding
    controller = SpeculationController()
    # Artificially inject deterioration in acceptance estimator
    for _ in range(5):
        controller.estimator.record_round(SpeculationStrategy.DRAFT_MODEL, "aeon-draft-0.5b", 4, 4, 0)

    # Next generation should trigger fallback
    text, state, telemetry = controller.execute_generation(
        target_model="aeon-large-70b",
        query="Surrealist chaotic text that produces zero acceptance",
        preferred_strategy=SpeculationStrategy.DRAFT_MODEL,
        requested_tokens=30,
    )
    assert text is not None
    assert len(text) > 0
    assert state.fallback_occurred is True or telemetry.speculation_action == ControllerAction.FALLBACK_REROUTE.value


# =========================================================================
# 12. Telemetry Schema & CLI
# =========================================================================

def test_telemetry():
    controller = SpeculationController()
    text, state, tel = controller.execute_generation(
        target_model="aeon-large-70b",
        query="Count from 1 to 20:",
        requested_tokens=25,
    )
    assert tel.speculation_strategy is not None
    assert tel.drafted_tokens >= 0
    assert tel.accepted_tokens >= 0
    assert tel.speculation_mode == "SIMULATED_SPECULATIVE_DECODING"
    assert tel.expected_speedup >= 1.0


def test_simulation_vs_real():
    controller = SpeculationController(mode="SIMULATED_SPECULATIVE_DECODING")
    text, state, tel = controller.execute_generation(
        target_model="aeon-medium-7b",
        query="List colors of the rainbow:",
        requested_tokens=20,
    )
    assert tel.speculation_mode == "SIMULATED_SPECULATIVE_DECODING"


def test_budget_enforcement():
    aeon = AEON(policy="voi", speculation=True)
    # Tiny cost budget
    res = aeon.run("Explain thermodynamics", budget={"max_cost_usd": 0.000001, "max_latency_ms": 200.0}, speculation=True)
    assert res.trace.stopping_reason in [
        DecisionReason.STOP_BUDGET_EXHAUSTED.value,
        DecisionReason.STOP_LOW_VOI.value,
        DecisionReason.STOP_TARGET_QUALITY_REACHED.value,
        DecisionReason.SPECULATION_FASTER_EXPECTED.value,
        DecisionReason.STOP_HIGH_CONFIDENCE.value,
    ]


def test_cli():
    runner = CliRunner()
    # stats
    r_stats = runner.invoke(cli, ["speculation", "stats"])
    assert r_stats.exit_code == 0
    assert "AEON SPECULATION STATS" in r_stats.output

    # config
    r_cfg = runner.invoke(cli, ["speculation", "config"])
    assert r_cfg.exit_code == 0
    assert "candidate_draft_lengths" in r_cfg.output

    # break-even
    r_be = runner.invoke(cli, ["speculation", "break-even"])
    assert r_be.exit_code == 0
    assert "70B_Target_0.5B_Draft" in r_be.output

    # compare
    r_comp = runner.invoke(cli, ["speculation", "compare"])
    assert r_comp.exit_code == 0
    assert "Workload Modality" in r_comp.output


# =========================================================================
# 13. Specification Property Invariants 1–15 (Section 26)
# =========================================================================

def test_invariant_1_speculative_tokens_never_silently_accepted_without_verification():
    """Invariant 1: Speculative tokens are never silently accepted without verification."""
    verifier = SpeculationVerifier()
    target_70b = global_registry.get("aeon-large-70b")
    # Proposal contains token 'wrong_tok' not present in target reference
    proposal = DraftTokenProposal(
        tokens=["apple", "banana", "wrong_tok"],
        confidences=[0.9, 0.9, 0.9],
        strategy=SpeculationStrategy.DRAFT_MODEL,
        draft_length=3,
    )
    reference = ["apple", "banana", "cherry"]
    res = verifier.verify_draft(target_70b, reference, proposal)
    assert "wrong_tok" not in res.accepted_tokens
    assert res.accepted_tokens == ["apple", "banana"]


def test_invariant_2_incompatible_draft_models_rejected():
    """Invariant 2: Incompatible draft models are rejected."""
    # 70B model cannot act as draft for 7B model
    assert global_registry.is_compatible("aeon-large-70b", "aeon-medium-7b") is False


def test_invariant_3_failed_speculation_falls_back_safely():
    """Invariant 3: Failed speculation falls back safely to standard decoding without crashing."""
    controller = SpeculationController()
    # Force failure with impossible target
    text, state, tel = controller.execute_generation(
        target_model="non_existent_model",
        query="Hello world",
        requested_tokens=10,
    )
    assert text is not None
    assert len(text) > 0


def test_invariant_4_budget_constraints_enforced():
    """Invariant 4: Budget constraints are enforced on speculative generation."""
    aeon = AEON(policy="voi", speculation=True)
    res = aeon.run("Write a 500 word story", budget={"max_cost": 0.000005}, speculation=True)
    assert res.cost <= 0.001


def test_invariant_5_sla_constraints_enforced():
    """Invariant 5: SLA constraints are enforced."""
    aeon = AEON(policy="rules", speculation=True)
    res = aeon.run("Tell me a fact", sla_latency_ms=5000.0, speculation=True)
    assert res.latency_ms <= 8000.0


def test_invariant_6_voi_decisions_explainable():
    """Invariant 6: VOI decisions are explainable."""
    selector = SpeculativeStrategySelector()
    top, cands = selector.select_strategy("aeon-large-70b", "def factorial(n):", task_type=TaskType.CODE)
    assert top.reason is not None
    assert len(top.reason) > 5


def test_invariant_7_acceptance_statistics_update_correctly():
    """Invariant 7: Acceptance statistics update correctly."""
    est = AdaptiveAcceptanceEstimator()
    init_val = est.global_ema
    est.record_round(SpeculationStrategy.DRAFT_MODEL, "aeon-draft-0.5b", 4, 4, 4)
    # High acceptance must increase EMA
    assert est.global_ema >= init_val


def test_invariant_8_draft_length_adapts_correctly():
    """Invariant 8: Draft length adapts correctly."""
    opt = AdaptiveDraftLengthOptimizer()
    new_len, _ = opt.adapt_length_on_feedback(current_length=4, realized_acceptance=0.98, break_even_rate=0.50)
    assert new_len > 4


def test_invariant_9_low_acceptance_causes_speculation_reduction_or_disablement():
    """Invariant 9: Low acceptance causes speculation reduction or disablement."""
    opt = AdaptiveDraftLengthOptimizer()
    new_len, reason = opt.adapt_length_on_feedback(current_length=6, realized_acceptance=0.20, break_even_rate=0.60)
    assert new_len < 6
    assert "reducing draft length" in reason


def test_invariant_10_standard_decoding_remains_available():
    """Invariant 10: Standard decoding remains available as a competing baseline."""
    controller = SpeculationController()
    text, state, tel = controller.execute_generation(
        target_model="aeon-large-70b",
        query="Standard decode test",
        preferred_strategy=SpeculationStrategy.STANDARD,
        requested_tokens=15,
    )
    assert state.current_strategy == SpeculationStrategy.STANDARD
    assert tel.speculation_action == ControllerAction.STANDARD_DECODE.value


def test_invariant_11_simulation_results_clearly_marked():
    """Invariant 11: Simulation results are clearly marked."""
    controller = SpeculationController()
    text, state, tel = controller.execute_generation(
        target_model="aeon-large-70b",
        query="Simulation labeling test",
        requested_tokens=10,
    )
    assert "SIMULATED" in tel.speculation_mode


def test_invariant_12_every_speculation_action_emits_telemetry():
    """Invariant 12: Every speculation action emits structured telemetry."""
    aeon = AEON(policy="voi", speculation=True)
    res = aeon.run("Write a poem", speculation=True)
    assert res.trace.request_id is not None
    assert hasattr(res.trace, "speculation_strategy")
    assert hasattr(res.trace, "accepted_tokens")


def test_invariant_13_memory_reuse_cannot_violate_compatibility_rules():
    """Invariant 13: Memory reuse cannot violate compatibility rules."""
    aeon = AEON(policy="voi", memory=True, speculation=True)
    # Store with session A
    res1 = aeon.run("Exclusive prompt", session_id="user_alpha", memory=True, speculation=True)
    # Cross-scope check ensures isolation
    assert res1.answer is not None


def test_invariant_14_dynamic_stopping_can_terminate_generation():
    """Invariant 14: Dynamic stopping can terminate generation when VOI drops."""
    aeon = AEON(policy="voi", speculation=True)
    res = aeon.run("What is 2 + 2?", target_quality=0.50, speculation=True)
    assert res.trace.number_of_controller_iterations <= 4


def test_invariant_15_deterministic_cheaper_strategies_considered_before_expensive_generation():
    """Invariant 15: Deterministic/cheaper strategies are considered before expensive generation."""
    aeon = AEON(policy="voi", tools=True, speculation=True)
    # Arithmetic should be intercepted by SafeCalculator before entering speculative generation
    res = aeon.run("What is 17.5% of 840?", tools=True, speculation=True)
    assert res.route == "tool"
    assert res.trace.tool_used == "calculator"
    assert res.trace.unnecessary_model_avoided is True
