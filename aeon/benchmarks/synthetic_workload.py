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


def generate_repeated_workload(count: int = 50) -> List[InferenceRequest]:
    """Workload A: Repeated exact requests (Section 23A)."""
    base_query = "What is the capital of France and what is its primary river?"
    return [
        InferenceRequest(
            query=base_query,
            ground_truth="Paris, Seine",
            task_hint="factual",
            memory=True,
        )
        for _ in range(count)
    ]


def generate_shared_prefix_workload(count: int = 20) -> List[InferenceRequest]:
    """Workload B: Shared-prefix workload with identical prompt headers (Section 23B)."""
    system_prefix = (
        "You are an expert financial analyst certified in quantitative valuation. "
        "Analyze the provided corporate metrics using standard DCF and DuPont models. "
    )
    user_questions = [
        "Calculate ROE given net income $5M and equity $25M.",
        "Analyze the current ratio of 1.8 with $10M in current assets.",
        "Compute debt-to-equity ratio if total liabilities are $40M and equity is $20M.",
        "Estimate operating margin from EBIT of $3M on revenue of $15M.",
        "What does an inventory turnover ratio of 6.0 indicate?",
    ]
    workload = []
    for i in range(count):
        q = user_questions[i % len(user_questions)]
        workload.append(
            InferenceRequest(
                query=f"{system_prefix}{q}",
                ground_truth="financial calculation",
                task_hint="math",
                memory=True,
            )
        )
    return workload


def generate_semantic_repetition_workload() -> List[List[InferenceRequest]]:
    """Workload C: Different wording, same underlying question (Section 23C)."""
    pairs = [
        [
            InferenceRequest(query="What is India's GDP in current US dollars?", ground_truth="$3.75T", task_hint="factual"),
            InferenceRequest(query="How large is the Indian economy in dollars?", ground_truth="$3.75T", task_hint="factual"),
        ],
        [
            InferenceRequest(query="What is the speed of light in vacuum?", ground_truth="299,792,458 m/s", task_hint="factual"),
            InferenceRequest(query="How fast does light travel in empty space?", ground_truth="299,792,458 m/s", task_hint="factual"),
        ],
        [
            InferenceRequest(query="Who developed the theory of general relativity?", ground_truth="Albert Einstein", task_hint="factual"),
            InferenceRequest(query="Which physicist formulated general relativity?", ground_truth="Albert Einstein", task_hint="factual"),
        ],
    ]
    return pairs


def generate_temporal_workload() -> List[InferenceRequest]:
    """Workload D: Frequently changing / time-sensitive information (Section 23D)."""
    return [
        InferenceRequest(query="What is the current stock price of TSLA right now?", task_hint="factual", consistency="STRICT"),
        InferenceRequest(query="What is today's headline news in Tokyo?", task_hint="factual", consistency="STRICT"),
        InferenceRequest(query="What is the current temperature in London right now?", task_hint="factual", consistency="STRICT"),
    ]


def generate_false_positive_pairs() -> List[Dict[str, Any]]:
    """
    Dedicated pairs for Section 27: Semantic cache false-positive experiment.
    Similar phrasing or high token overlap, but critically DIFFERENT semantics/years/entities!
    """
    return [
        {
            "query_a": "What was Apple's revenue in 2024?",
            "query_b": "What will Apple's revenue be in 2026?",
            "is_semantically_equivalent": False,
            "reason": "Temporal mismatch: 2024 historical vs 2026 forecast",
        },
        {
            "query_a": "Who is the CEO of Microsoft?",
            "query_b": "Who was the CEO of Microsoft in 2015?",
            "is_semantically_equivalent": False,
            "reason": "Temporal mismatch: current CEO vs 2015 CEO",
        },
        {
            "query_a": "What is the capital of Australia?",
            "query_b": "What is the capital of Austria?",
            "is_semantically_equivalent": False,
            "reason": "Entity mismatch: Australia (Canberra) vs Austria (Vienna)",
        },
        {
            "query_a": "Calculate 15 * 12",
            "query_b": "Calculate 15 * 18",
            "is_semantically_equivalent": False,
            "reason": "Numerical argument discrepancy",
        },
        {
            "query_a": "What is the capital of France?",
            "query_b": "What is the capital city of France?",
            "is_semantically_equivalent": True,
            "reason": "True semantic equivalence with near-identical phrasing",
        },
    ]


def generate_phase3_mixed_workload(count: int = 30) -> List[InferenceRequest]:
    """Workload G: Mixed workload combining easy, hard, repeated, semantic, prefix-heavy (Section 23G)."""
    prefix = "Act as an intelligent compute assistant. Answer with precision: "
    queries = [
        InferenceRequest(query=f"{prefix}What is 2 + 2?", ground_truth="4", task_hint="math", memory=True),
        InferenceRequest(query=f"{prefix}What is 2 + 2?", ground_truth="4", task_hint="math", memory=True),  # exact repeat
        InferenceRequest(query=f"{prefix}Calculate two plus two", ground_truth="4", task_hint="math", memory=True),  # semantic repeat
        InferenceRequest(query=f"{prefix}What is the capital of Canada?", ground_truth="Ottawa", task_hint="factual", memory=True),
        InferenceRequest(query=f"{prefix}What is the capital city of Canada?", ground_truth="Ottawa", task_hint="factual", memory=True),
        InferenceRequest(query=f"{prefix}Solve step-by-step: integral of 3x^2 dx", ground_truth="x^3 + C", task_hint="math", memory=True),
        InferenceRequest(query=f"{prefix}What was Tesla's quarterly production in 2024?", ground_truth="469k", task_hint="factual", consistency="FRESH", memory=True),
        InferenceRequest(query=f"{prefix}What will Tesla's quarterly production be in 2027?", ground_truth="Projection", task_hint="factual", consistency="STRICT", memory=True),
    ]
    out = []
    for i in range(count):
        out.append(queries[i % len(queries)].model_copy(deep=True))
    return out

