"""
Phase 2 Unit and Integration Test Suite for AEON.
Covers VOI estimation, dynamic stopping, calibration, budget constraints, and telemetry.
"""

import pytest
from aeon import AEON, InferenceRequest
from aeon.types import (
    ComputeState,
    ControllerAction,
    DecisionReason,
    RouteChoice,
    ReasoningMode,
    TaskType,
    DifficultyLevel,
    BudgetConstraints,
    UtilityWeights,
)
from aeon.controller.voi_estimator import VOIEstimator
from aeon.controller.dynamic_stopping import DynamicStoppingController
from aeon.decision.calibration import CalibrationTracker, EmpiricalCalibrationStore
from aeon.benchmarks.synthetic_workload import generate_synthetic_workload
from aeon.benchmarks.harness import BenchmarkHarness


def test_voi_estimator_marginal_utility_and_diminishing_returns():
    estimator = VOIEstimator()
    
    # State with initial low-tier model on hard task
    state_hard = ComputeState(
        request_id="test-voi-1",
        task_type=TaskType.REASONING,
        difficulty=0.85,
        difficulty_level=DifficultyLevel.HARD,
        uncertainty=0.30,
        current_model="aeon-small-1.5b",
        current_route=RouteChoice.SMALL_MODEL,
        current_reasoning_mode=ReasoningMode.DIRECT,
        current_quality_estimate=0.45,
        confidence=0.55,
        iteration=1,
    )
    
    # Candidate evaluation should prefer escalating to medium or large model with explicit reasoning
    result = estimator.evaluate_candidates(state_hard)
    assert result.action in [ControllerAction.ESCALATE_MODEL, ControllerAction.INCREASE_REASONING]
    assert result.marginal_quality_gain > 0.0
    assert result.is_economically_viable is True
    assert result.reason in [DecisionReason.ESCALATE_HIGH_EXPECTED_GAIN, DecisionReason.CONTINUE_HIGH_VOI]

    # Diminishing returns test: at higher iteration, marginal gain must be lower
    state_iteration_4 = state_hard.model_copy(update={"iteration": 4, "current_quality_estimate": 0.88})
    result_diminished = estimator.estimate_action(
        state_iteration_4, ControllerAction.ESCALATE_MODEL, "aeon-large-70b", ReasoningMode.EXPLICIT
    )
    assert result_diminished.marginal_quality_gain < result.marginal_quality_gain


def test_high_confidence_stopping():
    estimator = VOIEstimator(high_confidence_stop_threshold=0.90)
    
    # Solved state with high confidence
    state_solved = ComputeState(
        request_id="test-solved",
        task_type=TaskType.CLASSIFICATION,
        difficulty=0.10,
        difficulty_level=DifficultyLevel.EASY,
        uncertainty=0.10,
        current_model="aeon-decision-jev",
        current_route=RouteChoice.DECISION_MODEL,
        current_reasoning_mode=ReasoningMode.NONE,
        current_quality_estimate=0.92,
        confidence=0.95,
        iteration=1,
    )
    
    result = estimator.evaluate_candidates(state_solved)
    assert result.action == ControllerAction.STOP
    assert result.reason == DecisionReason.STOP_HIGH_CONFIDENCE


def test_budget_exhaustion_stopping():
    estimator = VOIEstimator()
    
    # State with depleted monetary budget
    state_exhausted = ComputeState(
        request_id="test-exhausted",
        task_type=TaskType.REASONING,
        difficulty=0.90,
        difficulty_level=DifficultyLevel.HARD,
        uncertainty=0.40,
        current_model="aeon-small-1.5b",
        current_route=RouteChoice.SMALL_MODEL,
        current_reasoning_mode=ReasoningMode.DIRECT,
        current_quality_estimate=0.50,
        confidence=0.50,
        remaining_cost_budget=0.0,  # Zero budget left
        iteration=1,
    )
    
    result = estimator.evaluate_candidates(state_exhausted)
    assert result.action == ControllerAction.STOP
    assert result.reason == DecisionReason.STOP_BUDGET_EXHAUSTED


def test_target_quality_stopping():
    estimator = VOIEstimator()
    
    state_good = ComputeState(
        request_id="test-target",
        task_type=TaskType.FACTUAL,
        difficulty=0.30,
        difficulty_level=DifficultyLevel.EASY,
        uncertainty=0.10,
        current_model="aeon-small-1.5b",
        current_route=RouteChoice.SMALL_MODEL,
        current_reasoning_mode=ReasoningMode.DIRECT,
        current_quality_estimate=0.88,
        confidence=0.85,
        iteration=1,
    )
    
    result = estimator.evaluate_candidates(state_good, target_quality=0.85)
    assert result.action == ControllerAction.STOP
    assert result.reason == DecisionReason.STOP_TARGET_QUALITY_REACHED


def test_calibration_tracker_ece_and_brier():
    tracker = CalibrationTracker(num_bins=5)
    
    # Perfect calibration synthetic data
    for _ in range(50):
        tracker.record(0.90, True)
    for _ in range(10):
        tracker.record(0.90, False)
        
    metrics = tracker.compute_metrics()
    assert metrics.num_samples == 60
    assert 0.0 <= metrics.brier_score <= 1.0
    assert 0.0 <= metrics.expected_calibration_error <= 1.0


def test_dynamic_stopping_controller_e2e():
    controller = DynamicStoppingController()
    
    # Easy query: should stop quickly (<= 3 iterations) and terminate on VOI/confidence
    req_easy = InferenceRequest(query="Is this spam: free prize winner claim now")
    res_easy = controller.run(req_easy)
    assert res_easy.trace.number_of_controller_iterations <= 3
    assert res_easy.trace.stopping_reason in [
        DecisionReason.STOP_HIGH_CONFIDENCE.value,
        DecisionReason.STOP_LOW_VOI.value,
    ]

    # Hard query: should trigger VOI-guided escalation
    req_hard = InferenceRequest(
        query="Prove that if a graph G has n vertices then by contradiction it must be connected"
    )
    res_hard = controller.run(req_hard)
    assert len(res_hard.trace.trace_steps) >= 1
    assert res_hard.trace.stopping_reason is not None
    assert res_hard.trace.voi is not None


def test_aeon_api_policy_switching():
    # Test VOI policy
    aeon_voi = AEON(policy="voi")
    res_voi = aeon_voi.run("What is 10 + 20?")
    assert res_voi.trace.stopping_reason is not None

    # Test Rules policy (Phase 1 backwards compatibility)
    aeon_rules = AEON(policy="rules")
    res_rules = aeon_rules.run("What is 10 + 20?")
    assert res_rules.route in [
        RouteChoice.SMALL_MODEL.value,
        RouteChoice.MEDIUM_MODEL.value,
        RouteChoice.DECISION_MODEL.value,
    ]

    # Test Cascade policy
    aeon_cascade = AEON(policy="cascade")
    res_cascade = aeon_cascade.run("What is 10 + 20?")
    assert "cascade" in res_cascade.route or res_cascade.route in [
        RouteChoice.SMALL_MODEL.value,
        RouteChoice.MEDIUM_MODEL.value,
    ]


def test_synthetic_workload_generator():
    workload = generate_synthetic_workload(scale_factor=1)
    assert len(workload) >= 12
    tasks = {r.task_hint for r in workload}
    assert "classification" in tasks
    assert "factual" in tasks
    assert "math" in tasks
    assert "reasoning" in tasks


def test_phase2_benchmark_harness():
    harness = BenchmarkHarness()
    phase2_res = harness.run_phase2()
    
    assert "Dense LLM Baseline (70B)" in phase2_res
    assert "AEON Adaptive (Phase 1)" in phase2_res
    assert "AEON Phase 2 VOI" in phase2_res

    aeon_phase2 = phase2_res["AEON Phase 2 VOI"]
    dense = phase2_res["Dense LLM Baseline (70B)"]

    # Phase 2 must beat Dense LLM on Quality / Dollar
    assert aeon_phase2.quality_per_dollar > dense.quality_per_dollar
    # Phase 2 must achieve lower total cost
    assert aeon_phase2.total_cost_usd < dense.total_cost_usd


def test_target_quality_experiment():
    harness = BenchmarkHarness()
    exp_res = harness.run_target_quality_experiment(target_quality=0.85)
    
    assert "AEON Phase 2 (Target Quality)" in exp_res
    p2_data = exp_res["AEON Phase 2 (Target Quality)"]
    assert p2_data["achieved_quality"] >= 0.80
    assert p2_data["total_cost_usd"] < exp_res["Dense LLM Baseline (70B)"]["total_cost_usd"]


def test_ablation_study():
    harness = BenchmarkHarness()
    ablations = harness.run_ablation_study()
    
    assert "1. Full AEON Phase 2" in ablations
    assert "2. AEON - VOI" in ablations
    assert "3. AEON - Confidence" in ablations
    assert "4. AEON - Difficulty" in ablations
    assert "5. AEON - Cost Awareness" in ablations
    assert "6. AEON - Dynamic Stopping" in ablations
