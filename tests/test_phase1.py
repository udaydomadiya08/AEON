"""
Phase 1 Unit and Integration Test Suite for AEON.
"""

import pytest
from aeon import AEON, InferenceRequest
from aeon.types import TaskType, DifficultyLevel, RouteChoice, ReasoningMode
from aeon.routing.task_classifier import TaskClassifier
from aeon.controller.difficulty_estimator import DifficultyEstimator
from aeon.decision.decision_model import TypedDecisionModel
from aeon.models.model_registry import global_registry
from aeon.benchmarks.harness import BenchmarkHarness


def test_task_classifier():
    classifier = TaskClassifier()
    
    # 1. Classification
    task_type, diff, level, unc = classifier.classify("Classify the sentiment of this review as positive or negative")
    assert task_type == TaskType.CLASSIFICATION
    assert level == DifficultyLevel.EASY
    
    # 2. Math
    task_type, diff, level, unc = classifier.classify("Calculate 17.3% of 847 and solve for x")
    assert task_type == TaskType.MATH
    assert diff > 0.30

    # 3. Complex proof
    task_type, diff, level, unc = classifier.classify("Prove that if a graph G has n vertices then G must be connected by contradiction")
    assert task_type == TaskType.REASONING
    assert level in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]


def test_typed_decision_model():
    model = TypedDecisionModel()
    schema = {
        "route": ["small", "medium", "large"],
        "confidence": "float",
        "needs_tool": "bool",
        "needs_verification": "bool",
    }
    pred = model.predict("Calculate 5 + 12 and verify the exact result", schema)
    assert "route" in pred
    assert isinstance(pred["confidence"], float)
    assert pred["needs_tool"] is True
    assert pred["needs_verification"] is True


def test_aeon_end_to_end_routing():
    aeon = AEON()

    # Simple classification should route to decision model
    res_easy = aeon.run("Classify the sentiment of this tweet: I love this!")
    assert res_easy.route == RouteChoice.DECISION_MODEL.value
    assert res_easy.trace.reasoning_mode == ReasoningMode.NONE.value
    assert res_easy.trace.compute_estimate < 1e10  # Low compute

    # Difficult reasoning query should route to large model
    res_hard = aeon.run("Prove that if a graph G has n vertices then by contradiction it must be connected")
    assert res_hard.route == RouteChoice.LARGE_MODEL.value
    assert res_hard.trace.reasoning_mode in [ReasoningMode.EXPLICIT.value, ReasoningMode.ADAPTIVE.value]
    assert res_hard.trace.compute_estimate > 1e11  # High compute


def test_telemetry_schema_compliance():
    aeon = AEON()
    res = aeon.run("What is the capital of France?")
    trace = res.trace

    # Check required fields from Section 20
    assert trace.request_id.startswith("aeon-")
    assert trace.task_type is not None
    assert trace.route is not None
    assert trace.model is not None
    assert trace.reasoning_mode is not None
    assert trace.input_tokens > 0
    assert trace.output_tokens > 0
    assert trace.latency_ms >= 0.0
    assert trace.compute_estimate > 0.0
    assert trace.memory_mb > 0.0
    assert trace.estimated_cost >= 0.0
    assert trace.confidence >= 0.0
    assert trace.latency_type == "MEASURED"
    assert trace.cost_type == "ESTIMATED"


def test_cascade_routing_escalation():
    # Use cascade mode
    aeon_cascade = AEON(use_cascade=True)
    res = aeon_cascade.run("Prove that if G has n vertices then G must be connected")
    assert "cascade" in res.route or res.route == RouteChoice.LARGE_MODEL.value
    assert res.trace.model in [m.name for m in global_registry.list_models()]


def test_benchmark_harness():
    harness = BenchmarkHarness()
    results = harness.run_all()

    assert "Dense LLM Baseline (70B)" in results
    assert "Small Model Baseline (1.5B)" in results
    assert "Model Cascade (1B->70B)" in results
    assert "AEON Adaptive (Phase 1)" in results

    aeon_metrics = results["AEON Adaptive (Phase 1)"]
    dense_metrics = results["Dense LLM Baseline (70B)"]

    # AEON should achieve substantially higher Quality / Dollar than Dense LLM
    assert aeon_metrics.quality_per_dollar > dense_metrics.quality_per_dollar
    # AEON total cost should be significantly lower than Dense LLM
    assert aeon_metrics.total_cost_usd < dense_metrics.total_cost_usd
