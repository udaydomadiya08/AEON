"""
Phase 4 Unit, Integration, and Invariant Test Suite for AEON.

Tests tool abstractions, AST-safe calculator, restricted Python sandbox,
SymPy equations, retrieval, structured computation, candidate generator,
planner, tool VOI, dynamic stopping integration, failure recovery,
budgets, invariants 1 to 8, CLI, and benchmarks.
Conforms to AEON Specification Sections 40 & 41.
"""

import pytest
from aeon import AEON
from aeon.types import (
    InferenceRequest,
    TaskType,
    ControllerAction,
    DecisionReason,
    RouteChoice,
    BudgetConstraints,
)
from aeon.tools.types import (
    ToolType,
    ToolStatus,
    VerificationStatus,
    ToolRequest,
    ToolResult,
    ToolCost,
    ToolBudget,
)
from aeon.tools.calculator import SafeCalculator
from aeon.tools.python_tool import RestrictedPythonTool
from aeon.tools.sympy_tool import SymPyTool
from aeon.tools.retrieval_tool import RetrievalTool
from aeon.tools.structured_tool import StructuredComputationTool
from aeon.tools.verifier import ToolVerifier
from aeon.tools.candidate_generator import ToolCandidateGenerator, ComputationalCandidate
from aeon.tools.planner import ComputationPlanner
from aeon.tools.tool_controller import ToolController
from aeon.tools.registry import ToolRegistry
from aeon.benchmarks.harness import BenchmarkHarness


# =========================================================================
# 1. Calculator Tests
# =========================================================================

def test_calculator_basic():
    calc = SafeCalculator()
    res = calc.execute(ToolRequest(query="17.5% of 840"))
    assert res.status == ToolStatus.SUCCESS
    assert res.output in ["147", "147.0"]

    res2 = calc.execute(ToolRequest(query="Calculate 25 * 40 - 12"))
    assert res2.status == ToolStatus.SUCCESS
    assert res2.output == "988"


def test_calculator_safe_parser():
    calc = SafeCalculator()
    # Complex but safe arithmetic
    res = calc.execute(ToolRequest(query="sqrt(144) + pow(2, 4) + abs(-10)"))
    assert res.status == ToolStatus.SUCCESS
    assert float(res.output) == 38.0


def test_calculator_rejects_unsafe_input():
    calc = SafeCalculator()
    # Unsafe code execution attempts
    res = calc.execute(ToolRequest(query="__import__('os').system('ls')"))
    assert res.status == ToolStatus.INVALID_INPUT
    assert "forbidden" in res.error.lower() or "could not extract" in res.error.lower()

    # Division by zero
    res_div0 = calc.execute(ToolRequest(query="10 / 0"))
    assert res_div0.status == ToolStatus.EXECUTION_ERROR
    assert "division by zero" in res_div0.error.lower()


# =========================================================================
# 2. SymPy and Python Execution Tests
# =========================================================================

def test_sympy_equation():
    sympy_tool = SymPyTool()
    req = ToolRequest(query="Solve x^2 + 5x + 6 = 0")
    cap = sympy_tool.can_handle(req)
    assert cap is not None
    assert cap.tool_type == ToolType.SYMPY

    res = sympy_tool.execute(req)
    assert res.status == ToolStatus.SUCCESS
    assert "-3" in res.output and "-2" in res.output

    # Test calculus derivative
    res_diff = sympy_tool.execute(ToolRequest(query="derivative of x^3 + 2*x"))
    assert res_diff.status == ToolStatus.SUCCESS
    assert "3*x**2 + 2" in res_diff.output


def test_python_restricted_execution():
    py_tool = RestrictedPythonTool()
    assert py_tool.name == "python"

    # Safe factorial
    res = py_tool.execute(ToolRequest(query="calculate factorial of 5"))
    assert res.status == ToolStatus.SUCCESS
    assert res.output == "120"

    # Rejection of blacklisted module import
    res_unsafe = py_tool.execute(ToolRequest(query="", parameters={"code": "import subprocess\nsubprocess.run('ls')"}))
    assert res_unsafe.status == ToolStatus.INVALID_INPUT
    assert "forbidden module" in res_unsafe.error.lower()


# =========================================================================
# 3. Capability, Candidate Generation, and Economics
# =========================================================================

def test_tool_capability_detection():
    registry = ToolRegistry()
    calc = registry.get("calculator")
    sympy_t = registry.get("sympy")
    retrieval_t = registry.get("retrieval")

    assert calc.can_handle(ToolRequest(query="What is 15 * 30?")) is not None
    assert calc.can_handle(ToolRequest(query="Solve x^2 - 4 = 0")) is None

    assert sympy_t.can_handle(ToolRequest(query="Solve x^2 - 4 = 0")) is not None
    assert retrieval_t.can_handle(ToolRequest(query="What was NVIDIA revenue in FY2025?")) is not None


def test_tool_candidate_generation():
    gen = ToolCandidateGenerator()
    cands = gen.generate_candidates(
        query="What is 17.5% of 840?",
        task_type=TaskType.MATH,
        difficulty=0.1,
        uncertainty=0.05,
    )
    assert len(cands) >= 2
    # Calculator should be top candidate
    assert cands[0].source_name == "calculator"
    assert cands[0].net_utility > 0.95


def test_tool_cost_estimation():
    calc = SafeCalculator()
    cost = calc.estimate_cost(ToolRequest(query="2 + 2"))
    assert cost.cost_usd < 0.00001
    assert cost.latency_ms < 1.0


def test_tool_voi():
    from aeon.controller.voi_estimator import VOIEstimator
    from aeon.types import ComputeState, DifficultyLevel, RouteChoice, ReasoningMode
    voi_est = VOIEstimator()
    state = ComputeState(
        request_id="test",
        task_type=TaskType.MATH,
        difficulty=0.2,
        difficulty_level=DifficultyLevel.EASY,
        uncertainty=0.1,
        current_model="aeon-small-1.5b",
        current_route=RouteChoice.SMALL_MODEL,
        current_reasoning_mode=ReasoningMode.DIRECT,
        current_quality_estimate=0.70,
    )
    gen = ToolCandidateGenerator()
    cands = gen.generate_candidates("What is 17.5% of 840?", TaskType.MATH, 0.2, 0.1)
    tool_cand = cands[0]

    voi_res = voi_est.estimate_tool_action(state, tool_cand)
    assert voi_res.is_economically_viable is True
    assert voi_res.voi_ratio > 1000.0


def test_tool_selection():
    aeon = AEON(policy="voi", tools=True)
    res = aeon.run("What is 17.5% of 840?")
    assert res.route == "tool"
    assert res.trace.tool_used == "calculator"
    assert res.answer in ["147", "147.0"]


def test_memory_vs_tool_selection():
    # If already in memory with safe hit, memory avoids even tool invocation!
    aeon = AEON(policy="voi", memory=True, tools=True)
    # First query populates memory
    res1 = aeon.run("What is 17.5% of 840?")
    assert res1.route == "tool"

    # Second query should hit memory
    res2 = aeon.run("What is 17.5% of 840?")
    assert res2.trace.memory_hit is True
    assert res2.cost <= 1e-6
    assert res2.cost < res1.cost


def test_tool_vs_model_selection():
    aeon = AEON(policy="voi", tools=True)

    # Tool sufficient
    res_tool = aeon.run("What is 17.5% of 840?")
    assert res_tool.route == "tool"

    # Model required for formal proof / reasoning
    res_model = aeon.run("Prove that the square root of 2 is irrational.")
    assert res_model.route != "tool"
    assert res_model.trace.model in ["aeon-medium-7b", "aeon-large-70b"]


def test_verification():
    sympy_tool = SymPyTool()
    res = sympy_tool.execute(ToolRequest(query="Solve x^2 + 5x + 6 = 0"))
    ver = sympy_tool.verify(res)
    assert ver.status == VerificationStatus.VERIFIED
    assert ver.discrepancy_detected is False


def test_tool_failure_fallback():
    # Artificially inject tool failure and verify dynamic fallback
    harness = BenchmarkHarness()
    res = harness.run_tool_failure_experiment()
    assert res["fallback_succeeded"] is True
    assert res["fallback_route"] is not None


def test_timeout_fallback():
    py_tool = RestrictedPythonTool()
    # Code with sleep/infinite loop
    res = py_tool.execute(ToolRequest(query="", parameters={"code": "import time\ntime.sleep(5.0)"}, timeout_seconds=0.1))
    assert res.status == ToolStatus.TIMEOUT


def test_multi_step_plan():
    planner = ComputationPlanner()
    top_cand = ComputationalCandidate(
        action=ControllerAction.RETRIEVAL,
        source_name="retrieval",
        expected_quality=0.98,
        expected_latency_ms=15.0,
        expected_cost_usd=0.000030,
        expected_compute_flops=6e5,
        expected_memory_mb=8.0,
        expected_energy_joules=0.0004,
        expected_risk=0.02,
    )
    plan = planner.create_plan(
        query="What is the CAGR of revenue between 2020 and 2024?",
        task_type=TaskType.FINANCIAL,
        top_candidate=top_cand,
    )
    assert len(plan.steps) >= 2


def test_plan_cost():
    planner = ComputationPlanner()
    top_cand = ComputationalCandidate(
        action=ControllerAction.CALCULATOR,
        source_name="calculator",
        expected_quality=1.0,
        expected_latency_ms=0.08,
        expected_cost_usd=0.0000005,
        expected_compute_flops=5e4,
        expected_memory_mb=0.1,
        expected_energy_joules=0.000002,
        expected_risk=0.001,
    )
    plan = planner.create_plan("2 + 2", TaskType.MATH, top_cand)
    assert plan.estimated_cost_usd < 0.00001


def test_plan_optimization():
    planner = ComputationPlanner()
    top_cand = ComputationalCandidate(
        action=ControllerAction.CALCULATOR,
        source_name="calculator",
        expected_quality=1.0,
        expected_latency_ms=0.08,
        expected_cost_usd=0.0000005,
        expected_compute_flops=5e4,
        expected_memory_mb=0.1,
        expected_energy_joules=0.000002,
        expected_risk=0.001,
    )
    plan = planner.create_plan("What is 10 + 20?", TaskType.MATH, top_cand)
    # Plan should not include any downstream model step
    assert not any(s.action in [ControllerAction.ESCALATE_MODEL, ControllerAction.CONTINUE] for s in plan.steps)


def test_budget_enforcement():
    aeon = AEON(policy="voi", tools=True)
    # Extremely low budget: $0.000001 (below model cost, only tools possible)
    res = aeon.run("What is 17.5% of 840?", cost_budget_usd=0.000001)
    assert res.cost <= 0.000001
    assert res.route == "tool"


def test_sla_enforcement():
    aeon = AEON(policy="voi", tools=True)
    # 5ms SLA: only deterministic tools can meet this
    res = aeon.run("What is 17.5% of 840?", sla_latency_ms=5.0)
    assert res.latency_ms <= 5.0
    assert res.route == "tool"


def test_tool_telemetry():
    aeon = AEON(policy="voi", tools=True)
    res = aeon.run("What is 17.5% of 840?")
    t = res.trace
    assert t.tool_used == "calculator"
    assert t.tool_category == "deterministic"
    assert t.unnecessary_model_avoided is True
    assert t.tool_cost_usd > 0.0


def test_cli_tools():
    from click.testing import CliRunner
    from aeon.cli.main import cli
    runner = CliRunner()
    
    # 1. tools list
    res_list = runner.invoke(cli, ["tools", "list"])
    assert res_list.exit_code == 0
    assert "calculator" in res_list.output

    # 2. tools stats
    res_stats = runner.invoke(cli, ["tools", "stats"])
    assert res_stats.exit_code == 0
    assert "Total Tool Calls" in res_stats.output

    # 3. tools benchmark
    res_bm = runner.invoke(cli, ["tools", "benchmark"])
    assert res_bm.exit_code == 0
    assert "Unnecessary Model Invocation Rate" in res_bm.output


def test_phase4_benchmark():
    harness = BenchmarkHarness()
    results = harness.run_phase4()
    assert "5. AEON Phase 4 (Tools + Memory)" in results
    p4 = results["5. AEON Phase 4 (Tools + Memory)"]
    llm = results["1. LLM-only (70B)"]
    assert p4.quality_per_dollar > llm.quality_per_dollar


# =========================================================================
# 4. Property Invariant Tests (Section 41)
# =========================================================================

def test_invariant_1_never_execute_tool_that_capability_says_cannot_handle():
    """Invariant 1: AEON must never execute a tool that its capability model says cannot handle the task."""
    calc = SafeCalculator()
    req = ToolRequest(query="Prove that sqrt(2) is irrational")
    assert calc.can_handle(req) is None


def test_invariant_2_unsafe_python_execution_must_be_rejected():
    """Invariant 2: Unsafe Python execution must be rejected."""
    py_tool = RestrictedPythonTool()
    req = ToolRequest(query="", parameters={"code": "import os\nos.remove('/etc/passwd')"})
    res = py_tool.execute(req)
    assert res.status == ToolStatus.INVALID_INPUT
    assert "forbidden" in res.error.lower()


def test_invariant_3_failed_tool_must_not_terminate_request_if_alternatives_exist():
    """Invariant 3: A failed tool must not terminate the entire request if alternatives exist."""
    harness = BenchmarkHarness()
    tfe = harness.run_tool_failure_experiment()
    assert tfe["fallback_succeeded"] is True


def test_invariant_4_tool_execution_must_respect_budget_constraints():
    """Invariant 4: Tool execution must respect budget constraints."""
    aeon = AEON(policy="voi", tools=True)
    res = aeon.run("What is 17.5% of 840?", cost_budget_usd=0.000005)
    assert res.cost <= 0.000005


def test_invariant_5_every_tool_action_must_be_represented_in_telemetry():
    """Invariant 5: Every tool action must be represented in telemetry."""
    aeon = AEON(policy="voi", tools=True)
    res = aeon.run("What is 17.5% of 840?")
    assert res.trace.tool_used is not None
    assert res.trace.tool_category is not None
    assert res.trace.tool_latency_ms >= 0.0


def test_invariant_6_every_selected_action_must_have_explanation():
    """Invariant 6: Every selected action must have an explanation."""
    aeon = AEON(policy="voi", tools=True)
    res = aeon.run("What is 17.5% of 840?")
    assert res.trace.decision_reason is not None


def test_invariant_7_verification_never_silently_converts_failed_into_success():
    """Invariant 7: Tool verification must never silently convert a failed verification into success."""
    verifier = ToolVerifier()
    res = ToolResult(
        tool_type=ToolType.CALCULATOR,
        status=ToolStatus.EXECUTION_ERROR,
        output=None,
        error="Execution failure",
    )
    v_res = verifier.verify_result(SafeCalculator(), res, "2 + 2")
    assert v_res.status == VerificationStatus.VERIFICATION_FAILED
    assert v_res.discrepancy_detected is True


def test_invariant_8_deterministic_tool_preferred_over_model_when_cheaper_and_sufficient():
    """Invariant 8: A deterministic tool should be preferred over a model when it provides sufficient expected quality at lower utility cost."""
    gen = ToolCandidateGenerator()
    cands = gen.generate_candidates(
        query="What is 17.5% of 840?",
        task_type=TaskType.MATH,
        difficulty=0.1,
        uncertainty=0.05,
    )
    # The top candidate must be a tool, not a model
    assert cands[0].action == ControllerAction.CALCULATOR
    assert cands[0].net_utility > cands[-1].net_utility
