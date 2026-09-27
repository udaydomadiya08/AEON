"""
Synthetic Workload Generator for AEON.

Generates diverse query workloads with calibrated difficulties, uncertainty profiles,
and ground truths across all core task modalities.
Conforms to AEON Specification Section 19.
"""

from typing import List, Dict, Any, Optional
from aeon.types import InferenceRequest


def generate_synthetic_workload(scale_factor: int = 1) -> List[InferenceRequest]:
    """
    Generates a balanced test distribution spanning easy to hard across:
    - easy classification, hard classification
    - easy factual, hard factual
    - simple arithmetic, complex arithmetic
    - simple reasoning, hard reasoning
    - ambiguous and edge cases
    """
    base_queries = [
        # 1. Easy Classification (trivial, high confidence, low VOI for LLM)
        InferenceRequest(
            query="Classify sentiment: 'I love this product, it works great!'",
            ground_truth="positive",
            task_hint="classification",
        ),
        InferenceRequest(
            query="Is this spam: 'Claim your free lottery winnings now!'",
            ground_truth="spam",
            task_hint="classification",
        ),

        # 2. Hard Classification (subtle sentiment, sarcasm, ambiguous category)
        InferenceRequest(
            query="Classify the sentiment: 'Oh wonderful, another delay right when I needed to catch my flight.'",
            ground_truth="negative",
            task_hint="classification",
        ),
        InferenceRequest(
            query="Classify into Legal, Financial, or Medical: 'The force majeure clause indemnifies the underwriter against non-performance.'",
            ground_truth="Legal",
            task_hint="classification",
        ),

        # 3. Easy Factual (lookup, direct fact)
        InferenceRequest(
            query="What is the capital of Japan?",
            ground_truth="Tokyo",
            task_hint="factual",
        ),
        InferenceRequest(
            query="What is the chemical symbol for gold?",
            ground_truth="Au",
            task_hint="factual",
        ),

        # 4. Hard Factual (niche historical dates, specific scientific parameters)
        InferenceRequest(
            query="In what year was the Treaty of Westphalia signed, and which primary conflicts did it resolve?",
            ground_truth="1648",
            task_hint="factual",
        ),

        # 5. Simple Arithmetic (elementary math)
        InferenceRequest(
            query="What is 45 + 55?",
            ground_truth="100",
            task_hint="math",
        ),
        InferenceRequest(
            query="What is 15 percent of 200?",
            ground_truth="30",
            task_hint="math",
        ),

        # 6. Complex Arithmetic & Multi-Step Math (combinatorics, algebra, compound equations)
        InferenceRequest(
            query="If f(x) = 3x^2 - 4x + 7, find the derivative f'(x) and calculate the critical point.",
            ground_truth="f'(x) = 6x - 4, critical point x = 2/3",
            task_hint="math",
        ),
        InferenceRequest(
            query="Solve step-by-step: If a loan of $10,000 compounds annually at 5% interest, what is the total balance after 3 years?",
            ground_truth="11576.25",
            task_hint="math",
        ),

        # 7. Simple Reasoning (cause-and-effect, commonsense deduction)
        InferenceRequest(
            query="Why does a hot air balloon rise when the air inside it is heated?",
            ground_truth="heated air expands and becomes less dense than surrounding air",
            task_hint="reasoning",
        ),

        # 8. Hard Reasoning & Proofs (logic, graph theory, theoretical CS)
        InferenceRequest(
            query="Prove that if a graph G has n vertices and strictly more than (n-1)(n-2)/2 edges, then G must be connected.",
            ground_truth="connected graph theorem",
            task_hint="reasoning",
        ),
        InferenceRequest(
            query="Explain by deduction why the Halting Problem is undecidable using a diagonal contradiction argument.",
            ground_truth="diagonal contradiction",
            task_hint="reasoning",
        ),

        # 9. Ambiguous Queries (uncertainty testing)
        InferenceRequest(
            query="Perhaps maybe this could be better or worse, what do you think is the subjective verdict?",
            ground_truth="balanced subjective perspective",
            task_hint="reasoning",
        ),
    ]

    workload = []
    for _ in range(scale_factor):
        for req in base_queries:
            workload.append(req.model_copy(deep=True))
    return workload
