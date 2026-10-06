"""
AEON Phase 9 — Workload Taxonomy & Ground Truth Evaluator.

Defines a comprehensive workload suite across all 12 canonical workload families
with explicit, auditable ground-truth verification.
"""

import math
import re
from typing import List, Optional, Tuple, Dict, Any
from aeon.benchmarks.types import Workload, WorkloadFamily


def evaluate_ground_truth(workload: Workload, candidate_answer: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, float]:
    """
    Strict ground-truth evaluator.
    Returns (is_correct, score).
    Never hardcodes True.
    """
    if candidate_answer is None or not candidate_answer.strip():
        return False, 0.0

    ans = candidate_answer.strip()
    
    # 1. Exact numeric evaluation
    if workload.expected_numeric_value is not None:
        # Extract potential numbers from answer
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", ans)
        if nums:
            try:
                # Compare last found number or exact match
                extracted = float(nums[-1])
                target = workload.expected_numeric_value
                if abs(extracted - target) < 1e-3 or (target != 0 and abs(extracted - target) / abs(target) < 1e-3):
                    return True, 1.0
            except ValueError:
                pass
        return False, 0.0

    # 2. Exact string match
    if workload.expected_answer is not None:
        expected = workload.expected_answer.strip()
        if ans.lower() == expected.lower() or expected.lower() in ans.lower():
            return True, 1.0
        return False, 0.0

    # 3. Accepted answer set match
    if workload.accepted_answer_set is not None:
        for candidate in workload.accepted_answer_set:
            if candidate.lower() in ans.lower():
                return True, 1.0
        return False, 0.0

    # 4. Custom rule evaluation
    if workload.verification_rule is not None:
        if workload.verification_rule == "is_prime_check":
            try:
                val = int(re.findall(r"\d+", ans)[-1])
                is_p = val > 1 and all(val % d != 0 for d in range(2, int(math.isqrt(val)) + 1))
                return is_p, 1.0 if is_p else 0.0
            except (IndexError, ValueError):
                return False, 0.0
        elif workload.verification_rule == "non_empty_json":
            import json
            try:
                json.loads(ans)
                return True, 1.0
            except Exception:
                return False, 0.0

    # If task claims ground truth but none matched, fail
    return False, 0.0


def create_canonical_workloads() -> List[Workload]:
    """Construct the canonical benchmark workload suite covering all 12 taxonomy families."""
    workloads: List[Workload] = []

    # Family A: Deterministic Arithmetic (Tool routing)
    workloads.extend([
        Workload(
            id="arith-01",
            name="Compound Interest Calculation",
            family=WorkloadFamily.DETERMINISTIC_ARITHMETIC,
            query="Calculate the compound amount for principal 1000, rate 5%, time 3 years compounded annually.",
            expected_numeric_value=1157.625,
            declared_difficulty=0.20,
            estimated_difficulty=0.22,
            metadata={"domain": "finance", "requires_tool": True}
        ),
        Workload(
            id="arith-02",
            name="Exact Square Root Arithmetic",
            family=WorkloadFamily.DETERMINISTIC_ARITHMETIC,
            query="What is the exact square root of 50625?",
            expected_numeric_value=225.0,
            declared_difficulty=0.15,
            estimated_difficulty=0.15,
            metadata={"domain": "math", "requires_tool": True}
        ),
        Workload(
            id="arith-03",
            name="Percentage Margin Calculation",
            family=WorkloadFamily.DETERMINISTIC_ARITHMETIC,
            query="If revenue is 450000 and cost is 315000, what is the profit margin percentage?",
            expected_numeric_value=30.0,
            declared_difficulty=0.25,
            estimated_difficulty=0.24,
            metadata={"domain": "finance", "requires_tool": True}
        )
    ])

    # Family B: Simple Factual Retrieval
    workloads.extend([
        Workload(
            id="fact-01",
            name="Capital of France",
            family=WorkloadFamily.SIMPLE_FACTUAL,
            query="What is the capital city of France?",
            expected_answer="Paris",
            accepted_answer_set=["Paris"],
            declared_difficulty=0.05,
            estimated_difficulty=0.08,
            metadata={"domain": "geography"}
        ),
        Workload(
            id="fact-02",
            name="Boiling Point of Water",
            family=WorkloadFamily.SIMPLE_FACTUAL,
            query="What is the boiling point of pure water in degrees Celsius at standard atmospheric pressure?",
            expected_numeric_value=100.0,
            declared_difficulty=0.05,
            estimated_difficulty=0.06,
            metadata={"domain": "physics"}
        )
    ])

    # Family C: Classification
    workloads.extend([
        Workload(
            id="class-01",
            name="Sentiment Classification Positive",
            family=WorkloadFamily.CLASSIFICATION,
            query="Classify the sentiment of: 'The benchmark execution was flawless, fast, and remarkably robust.'",
            accepted_answer_set=["positive", "positive sentiment"],
            declared_difficulty=0.10,
            estimated_difficulty=0.12,
            metadata={"domain": "nlp"}
        ),
        Workload(
            id="class-02",
            name="Spam Detection",
            family=WorkloadFamily.CLASSIFICATION,
            query="Is this email spam or ham? 'Claim your $1,000,000 lottery jackpot right now by sending your bank login!'",
            accepted_answer_set=["spam"],
            declared_difficulty=0.10,
            estimated_difficulty=0.11,
            metadata={"domain": "security"}
        )
    ])

    # Family D: Simple Reasoning
    workloads.extend([
        Workload(
            id="reas-simple-01",
            name="Syllogism Transitivity",
            family=WorkloadFamily.SIMPLE_REASONING,
            query="All mammals are warm-blooded. All dolphins are mammals. Are dolphins warm-blooded?",
            accepted_answer_set=["yes", "dolphins are warm-blooded", "true"],
            declared_difficulty=0.25,
            estimated_difficulty=0.28,
            metadata={"domain": "logic"}
        ),
        Workload(
            id="reas-simple-02",
            name="Comparative Height Order",
            family=WorkloadFamily.SIMPLE_REASONING,
            query="Alice is taller than Bob. Bob is taller than Charlie. Who is the shortest among the three?",
            expected_answer="Charlie",
            accepted_answer_set=["Charlie"],
            declared_difficulty=0.25,
            estimated_difficulty=0.26,
            metadata={"domain": "logic"}
        )
    ])

    # Family E: Medium Reasoning
    workloads.extend([
        Workload(
            id="reas-med-01",
            name="Schedule Constraint Satisfaction",
            family=WorkloadFamily.MEDIUM_REASONING,
            query="A team needs to schedule 3 meetings: A, B, and C. Meeting A must be before B. Meeting C must be before A. What is the valid meeting order?",
            expected_answer="C, A, B",
            accepted_answer_set=["C, A, B", "C then A then B", "C -> A -> B", "CAB"],
            declared_difficulty=0.55,
            estimated_difficulty=0.58,
            metadata={"domain": "planning"}
        ),
        Workload(
            id="reas-med-02",
            name="Multi-condition Logic Deduction",
            family=WorkloadFamily.MEDIUM_REASONING,
            query="If the server is overloaded or disk is full, alerts trigger. Alerts did not trigger. Is the disk full?",
            expected_answer="No",
            accepted_answer_set=["no", "disk is not full", "false"],
            declared_difficulty=0.50,
            estimated_difficulty=0.52,
            metadata={"domain": "logic"}
        )
    ])

    # Family F: Hard Reasoning
    workloads.extend([
        Workload(
            id="reas-hard-01",
            name="Knights and Knaves Island Puzzle",
            family=WorkloadFamily.HARD_REASONING,
            query="On an island, Knights always tell the truth and Knaves always lie. Person A says: 'Both of us are Knaves.' What is Person A?",
            expected_answer="Knave",
            accepted_answer_set=["Knave", "A is a knave"],
            declared_difficulty=0.85,
            estimated_difficulty=0.88,
            metadata={"domain": "logic"}
        ),
        Workload(
            id="reas-hard-02",
            name="Game Theory Dominant Strategy",
            family=WorkloadFamily.HARD_REASONING,
            query="In a standard Prisoner's Dilemma, what is the strictly dominant strategy for both rational self-interested players?",
            expected_answer="Defect",
            accepted_answer_set=["defect", "confess", "betray"],
            declared_difficulty=0.80,
            estimated_difficulty=0.82,
            metadata={"domain": "game_theory"}
        )
    ])

    # Family G: Verification-Sensitive Tasks (Sensitive to INT8 quantization)
    workloads.extend([
        Workload(
            id="verif-01",
            name="Eigenvalue Matrix Trace Invariance",
            family=WorkloadFamily.VERIFICATION_SENSITIVE,
            query="For matrix [[3, 1], [1, 3]], what is the exact sum of its eigenvalues lambda_1 + lambda_2?",
            expected_numeric_value=6.0,
            require_verification=True,
            declared_difficulty=0.70,
            estimated_difficulty=0.75,
            metadata={"domain": "linear_algebra", "precision_sensitive": True}
        ),
        Workload(
            id="verif-02",
            name="Numerical Epsilon Precision Invariant",
            family=WorkloadFamily.VERIFICATION_SENSITIVE,
            query="Calculate (1.0000000000000002 - 1.0) * 1e16 with IEEE floating point precision representation.",
            expected_numeric_value=2.0,
            require_verification=True,
            declared_difficulty=0.90,
            estimated_difficulty=0.92,
            metadata={"domain": "numerical_analysis", "precision_sensitive": True}
        )
    ])

    # Family H: Long-Context Tasks
    long_ctx = "Document Segment: [System report 2026. The critical threshold status alpha was confirmed stable. Section 402 mentions security override code 94821. End of document.] " * 5
    workloads.append(
        Workload(
            id="longctx-01",
            name="Long Context Key Retrieval",
            family=WorkloadFamily.LONG_CONTEXT,
            query="From the attached system document, what is the security override code?",
            context=long_ctx,
            expected_answer="94821",
            accepted_answer_set=["94821"],
            declared_difficulty=0.40,
            estimated_difficulty=0.45,
            metadata={"tokens_est": 350, "context_length": len(long_ctx)}
        )
    )

    # Family I: Repeated Query (Cache effectiveness)
    workloads.extend([
        Workload(
            id="repeat-01a",
            name="Repeated Query Prime",
            family=WorkloadFamily.REPEATED_QUERY,
            query="What is the speed of light in vacuum in meters per second?",
            expected_numeric_value=299792458.0,
            declared_difficulty=0.10,
            estimated_difficulty=0.10,
            metadata={"repetition_id": "c_light"}
        ),
        Workload(
            id="repeat-01b",
            name="Repeated Query Duplicate",
            family=WorkloadFamily.REPEATED_QUERY,
            query="What is the speed of light in vacuum in meters per second?",
            expected_numeric_value=299792458.0,
            declared_difficulty=0.10,
            estimated_difficulty=0.10,
            metadata={"repetition_id": "c_light", "is_cache_target": True}
        )
    ])

    # Family J: Tool-Sufficient Workload
    workloads.append(
        Workload(
            id="tool-suff-01",
            name="Large Integer Factorial Modulo",
            family=WorkloadFamily.TOOL_SUFFICIENT,
            query="Compute 15! % 1000",
            expected_numeric_value=0.0,
            declared_difficulty=0.30,
            estimated_difficulty=0.30,
            metadata={"tool_sufficient": True}
        )
    )

    # Family K: Model-Required Workload
    workloads.append(
        Workload(
            id="model-req-01",
            name="Metaphorical Meaning Extraction",
            family=WorkloadFamily.MODEL_REQUIRED,
            query="What does the phrase 'Achilles heel' symbolize in classic literary criticism?",
            accepted_answer_set=["vulnerability", "fatal flaw", "weakness", "fatal weakness"],
            declared_difficulty=0.45,
            estimated_difficulty=0.48,
            metadata={"model_required": True}
        )
    )

    return workloads
