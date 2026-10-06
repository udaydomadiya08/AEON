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


# =========================================================================
# Phase 4 Tool-Aware Synthetic Workloads (Section 28 & 31)
# =========================================================================

def generate_arithmetic_workload(count: int = 20) -> List[InferenceRequest]:
    """Arithmetic workload (Section 28A): exact percentages and arithmetic calculations."""
    samples = [
        ("What is 17.5% of 840?", "147"),
        ("Calculate 25 * 40 - 12", "988"),
        ("Compute 144 / 12 + 35 * 2", "82"),
        ("What is 8.25% of 2400?", "198"),
        ("Evaluate 2 ** 8 + 15 * 4", "316"),
        ("What is 33.3% of 900?", "299.7"),
        ("Calculate (50 + 25) * 4 / 2", "150"),
        ("What is 12.5% of 1600?", "200"),
    ]
    out = []
    for i in range(count):
        q, gt = samples[i % len(samples)]
        out.append(InferenceRequest(query=q, ground_truth=gt, task_hint="arithmetic", tools=True))
    return out


def generate_symbolic_math_workload(count: int = 15) -> List[InferenceRequest]:
    """Symbolic math workload (Section 28C): equations, derivatives, integrals, factorizations."""
    samples = [
        ("Solve x^2 + 5x + 6 = 0", "x = -3, -2"),
        ("Solve 2*x + 8 = 20", "x = 6"),
        ("Derivative of x^3 + 2*x", "3*x**2 + 2"),
        ("Integral of 3*x^2", "x**3"),
        ("Factorize x^2 - 9", "(x - 3)*(x + 3)"),
        ("Solve x^2 - 4 = 0", "x = -2, 2"),
    ]
    out = []
    for i in range(count):
        q, gt = samples[i % len(samples)]
        out.append(InferenceRequest(query=q, ground_truth=gt, task_hint="symbolic_math", tools=True))
    return out


def generate_financial_workload(count: int = 15) -> List[InferenceRequest]:
    """Financial calculations workload (Section 28B): CAGR, margins, returns."""
    samples = [
        ("Calculate CAGR from initial 100 to final 250 over 5 years", "20.11%"),
        ("Calculate profit margin with revenue 1000 and cost 650", "35.00%"),
        ("What is the CAGR for revenue starting at 50 and ending at 120 in 3 years", "33.89%"),
        ("Calculate gross margin with revenue 5000 and cost 2000", "60.00%"),
    ]
    out = []
    for i in range(count):
        q, gt = samples[i % len(samples)]
        out.append(InferenceRequest(query=q, ground_truth=gt, task_hint="financial", tools=True))
    return out


def generate_retrieval_workload(count: int = 15) -> List[InferenceRequest]:
    """Retrieval workload (Section 28E): fact lookup and source attribution."""
    samples = [
        ("What was NVIDIA revenue in FY2025?", "$130.5 billion"),
        ("What was Apple revenue in 2024?", "$391.04 billion"),
        ("What is the GDP of India?", "$3.94 trillion"),
        ("What is the capital of France?", "Paris"),
    ]
    out = []
    for i in range(count):
        q, gt = samples[i % len(samples)]
        out.append(InferenceRequest(query=q, ground_truth=gt, task_hint="factual", tools=True, memory=True))
    return out


def generate_reasoning_workload(count: int = 15) -> List[InferenceRequest]:
    """Formal and deductive reasoning workload where models are required."""
    samples = [
        ("Prove that the square root of 2 is irrational.", "Proof by contradiction"),
        ("Explain why all prime numbers greater than 2 must be odd.", "Deduction on divisibility by 2"),
        ("Deduce the truth value: If every graph with n vertices has degree d, does G contain a cycle?", "Cycle existence proof"),
        ("Analyze the strategic trade-offs of microservices vs monolithic architecture in high-throughput systems.", "Trade-off evaluation"),
    ]
    out = []
    for i in range(count):
        q, gt = samples[i % len(samples)]
        out.append(InferenceRequest(query=q, ground_truth=gt, task_hint="reasoning", tools=True))
    return out


def generate_model_vs_tool_workload() -> List[Dict[str, Any]]:
    """
    Dedicated test cases for Section 31: Model vs Tool Experiment.
    Contrasts tasks where a deterministic tool is sufficient against tasks where model reasoning is necessary.
    """
    return [
        {"query": "What is 17.5% of 840?", "task_category": "arithmetic", "tool_sufficient": True, "expected_route": "tool"},
        {"query": "Solve x^2 + 5x + 6 = 0", "task_category": "symbolic_math", "tool_sufficient": True, "expected_route": "tool"},
        {"query": "Calculate CAGR from initial 100 to final 250 over 5 years", "task_category": "financial", "tool_sufficient": True, "expected_route": "tool"},
        {"query": "What was NVIDIA revenue in FY2025?", "task_category": "retrieval", "tool_sufficient": True, "expected_route": "tool"},
        {"query": "Calculate mean of 12, 18, 24, 30", "task_category": "data_analysis", "tool_sufficient": True, "expected_route": "tool"},
        {"query": "Prove that the square root of 2 is irrational.", "task_category": "reasoning", "tool_sufficient": False, "expected_route": "model"},
        {"query": "Write a persuasive essay analyzing the ethical implications of autonomous AI agents.", "task_category": "creative", "tool_sufficient": False, "expected_route": "model"},
        {"query": "Analyze the philosophical distinction between epistemic and aleatoric uncertainty.", "task_category": "reasoning", "tool_sufficient": False, "expected_route": "model"},
    ]


def generate_phase4_mixed_workload(count: int = 30) -> List[InferenceRequest]:
    """Section 28G: Comprehensive Phase 4 Mixed Workload."""
    queries = [
        InferenceRequest(query="What is 17.5% of 840?", ground_truth="147", task_hint="arithmetic", tools=True, memory=True),
        InferenceRequest(query="Solve x^2 + 5x + 6 = 0", ground_truth="x = -3, -2", task_hint="symbolic_math", tools=True, memory=True),
        InferenceRequest(query="Calculate CAGR from initial 100 to final 250 over 5 years", ground_truth="20.11%", task_hint="financial", tools=True, memory=True),
        InferenceRequest(query="What was NVIDIA revenue in FY2025?", ground_truth="$130.5 billion", task_hint="factual", tools=True, memory=True),
        InferenceRequest(query="Prove that the square root of 2 is irrational.", ground_truth="Proof", task_hint="reasoning", tools=True, memory=True),
        InferenceRequest(query="Calculate 25 * 40 - 12", ground_truth="988", task_hint="arithmetic", tools=True, memory=True),
        InferenceRequest(query="What is 17.5% of 840?", ground_truth="147", task_hint="arithmetic", tools=True, memory=True),  # repeat to test memory + tool synergy
    ]
    out = []
    for i in range(count):
        out.append(queries[i % len(queries)].model_copy(deep=True))
    return out


def generate_phase5_speculation_workload() -> Dict[str, List[InferenceRequest]]:
    """
    Dedicated Phase 5 Benchmark Workloads conforming to Section 21:
    A. Highly predictable generation (High speculation speedup)
    B. Normal natural-language generation (Moderate benefit)
    C. Difficult reasoning (Low speculation benefit)
    D. Code generation (Structured syntactic patterns)
    E. Repetitive generation (N-gram optimal)
    F. Short outputs (Speculation overhead penalty)
    G. Long outputs (Compounded speedup)
    H. Low acceptance workload (Verifies AEON disables speculation)
    """
    return {
        "predictable": [
            InferenceRequest(query="List the months of the year in order from January to December:", task_hint="factual", speculation=True),
            InferenceRequest(query="Count from 1 to 20 in sequential order with commas:", task_hint="factual", speculation=True),
            InferenceRequest(query="Output the standard English alphabet letters from A to Z:", task_hint="factual", speculation=True),
        ],
        "natural_language": [
            InferenceRequest(query="Describe the history and architectural beauty of the city of Kyoto.", task_hint="creative", speculation=True),
            InferenceRequest(query="Summarize the primary causes of the Industrial Revolution in Europe.", task_hint="factual", speculation=True),
            InferenceRequest(query="Explain the biological process of photosynthesis in plants.", task_hint="factual", speculation=True),
        ],
        "difficult_reasoning": [
            InferenceRequest(query="Deduce whether P equals NP and prove why polynomial time reduction fails.", task_hint="reasoning", speculation=True),
            InferenceRequest(query="Solve this adversarial logic riddle involving 3 truth-tellers and 2 liars.", task_hint="reasoning", speculation=True),
            InferenceRequest(query="Construct a formal topological proof of the Brouwer fixed-point theorem.", task_hint="reasoning", speculation=True),
        ],
        "code_generation": [
            InferenceRequest(query="def binary_search(arr, target): Write standard Python implementation with boundary checks.", task_hint="code", speculation=True),
            InferenceRequest(query="class LRUCache: Implement get and put methods in Python with doubly linked list.", task_hint="code", speculation=True),
            InferenceRequest(query="Write a SQL query selecting top 5 customers by total order amount grouped by year.", task_hint="code", speculation=True),
        ],
        "repetitive": [
            InferenceRequest(query="Echo the following configuration block 3 times: [Server: host=localhost, port=8080, protocol=http, timeout=30s]", task_hint="extraction", speculation=True),
            InferenceRequest(query="Repeat the word 'ACCEPTED' 20 times separated by hyphens:", task_hint="factual", speculation=True),
            InferenceRequest(query="Repeat schema: user_id, user_name, user_email, user_role, user_status across 5 records.", task_hint="code", speculation=True),
        ],
        "short_output": [
            InferenceRequest(query="Answer with one word: What color is the clear daytime sky?", task_hint="factual", speculation=True),
            InferenceRequest(query="Is 17 a prime number? Answer Yes or No.", task_hint="classification", speculation=True),
        ],
        "long_output": [
            InferenceRequest(query="Write an extensive 200-word analysis on the economic impact of global semiconductor supply chains.", task_hint="creative", speculation=True),
            InferenceRequest(query="Generate a complete end-to-end technical tutorial on building microservices in Python.", task_hint="code", speculation=True),
        ],
        "low_acceptance": [
            InferenceRequest(query="Generate an unpredictable sequence of arbitrary random characters and symbols: #!9*kQ~", task_hint="creative", speculation=True),
            InferenceRequest(query="Write a chaotic surrealist non-sequitur sentence where every word is unlinked.", task_hint="creative", speculation=True),
        ],
    }


def generate_phase6_reasoning_workload() -> Dict[str, List[InferenceRequest]]:
    """
    AEON Phase 6 — Adaptive Latent & Token-Efficient Reasoning Workload Suite.
    Categorizes tasks across 6 distinct computational reasoning modalities:
    1. No Reasoning (Direct Lookup / Classification)
    2. Shallow Reasoning (Simple Deductions, 1-2 Steps)
    3. Medium Reasoning (Multi-Step Logic & Optimization)
    4. Deep Reasoning (Formal Proofs, Combinatorial Logic, Adversarial Riddles)
    5. Tool-Assisted Reasoning (Hybrid Symbolic/Deterministic Verification)
    6. Budget-Constrained Reasoning (Tight Latency/Cost SLAs Testing Early Termination)
    """
    return {
        "no_reasoning": [
            InferenceRequest(query="Classify sentiment: 'This product exceeded all my expectations!'", ground_truth="positive", task_hint="classification", reasoning=True),
            InferenceRequest(query="What is the capital city of France?", ground_truth="Paris", task_hint="factual", reasoning=True),
            InferenceRequest(query="Extract the date from: Meeting confirmed for October 14, 2026 at 2 PM.", ground_truth="October 14, 2026", task_hint="extraction", reasoning=True),
        ],
        "shallow_reasoning": [
            InferenceRequest(query="If Alice is older than Bob, and Bob is older than Charlie, is Alice older than Charlie?", ground_truth="Yes", task_hint="reasoning", reasoning=True),
            InferenceRequest(query="A train leaves at 3:15 PM and arrives at 4:45 PM. How many minutes did the journey take?", ground_truth="90", task_hint="reasoning", reasoning=True),
            InferenceRequest(query="If all roses are flowers and all flowers need sunlight, do roses need sunlight?", ground_truth="Yes", task_hint="reasoning", reasoning=True),
        ],
        "medium_reasoning": [
            InferenceRequest(query="Determine whether a graph with vertices {A, B, C, D} and edges (A,B), (B,C), (C,D), (D,A) is bipartite.", ground_truth="Yes, it is bipartite", task_hint="reasoning", reasoning=True),
            InferenceRequest(query="Optimize schedule: Job A takes 3h, Job B takes 2h, Job C takes 4h. Job C depends on A and B. What is the minimum completion time with 2 workers?", ground_truth="7 hours", task_hint="reasoning", reasoning=True),
            InferenceRequest(query="Given two sorted arrays of size m and n, analyze the time complexity of finding the median in O(log(min(m, n))).", ground_truth="O(log(min(m, n)))", task_hint="reasoning", reasoning=True),
        ],
        "deep_reasoning": [
            InferenceRequest(query="Construct a formal deductive proof that there are infinitely many prime numbers using Euclid's construction.", ground_truth="Infinitely many primes proof", task_hint="reasoning", reasoning=True),
            InferenceRequest(query="Solve this logic puzzle: Three inhabitants A, B, and C on the Island of Knights and Knaves. A says 'B is a knave'. B says 'A and C are of the same type'. What is C?", ground_truth="C is a knight", task_hint="reasoning", reasoning=True),
            InferenceRequest(query="Analyze whether the Halting Problem can be solved for a Turing machine with bounded tape length L in finite time.", ground_truth="Decidable in finite time because states are finite", task_hint="reasoning", reasoning=True),
        ],
        "tool_assisted": [
            InferenceRequest(query="Calculate exactly: 47291 * 83921 + 94812 / 12 and verify deterministically.", ground_truth="3968708911", task_hint="arithmetic", tools=True, reasoning=True),
            InferenceRequest(query="Solve symbolic system: 3*x + 4*y = 25 and 2*x - y = 9 for (x, y).", ground_truth="x=5.545, y=2.09", task_hint="symbolic_math", tools=True, reasoning=True),
        ],
        "budget_constrained": [
            InferenceRequest(query="Evaluate whether P vs NP can be settled by relativizing techniques, under tight budget constraint.", task_hint="reasoning", sla_latency_ms=150.0, cost_budget_usd=0.0005, reasoning=True),
            InferenceRequest(query="Find the optimal Nash equilibrium in a complex 5-player non-zero-sum game under 100ms SLA.", task_hint="reasoning", sla_latency_ms=100.0, cost_budget_usd=0.0002, reasoning=True),
        ],
    }


def generate_phase7_depth_workload() -> Dict[str, List[InferenceRequest]]:
    """
    AEON Phase 7 — Adaptive Model Depth & Mixture-of-Depths Workload Suite.
    Categorizes tasks across 12 distinct computational depth & routing modalities:
    1. trivial (Immediate early exit, min layers)
    2. easy factual (Shallow layers, quick exit)
    3. arithmetic (Deterministic tool check, early exit if tool handled)
    4. medium reasoning (Moderate layers, selective token depth)
    5. hard reasoning (Deep layers, high token capacity)
    6. proof/logic (Max depth, strict verification gate)
    7. code (Algorithmic syntax, moderate-to-deep allocation)
    8. long-context (Token-selective MoD allocation, budget preservation)
    9. repetitive (Sparse MoD computation, high layer skipping)
    10. uncertainty-heavy (Depth scaling to resolve entropy)
    11. verification-heavy (High confidence required before exit)
    12. mixed workloads (Multi-modal depth optimization)
    """
    return {
        "trivial": [
            InferenceRequest(query="Hi, how are you?", ground_truth="Hello", task_hint="classification", adaptive_depth=True),
            InferenceRequest(query="Say yes.", ground_truth="yes", task_hint="factual", adaptive_depth=True),
            InferenceRequest(query="Echo 'pong'", ground_truth="pong", task_hint="extraction", adaptive_depth=True),
        ],
        "easy_factual": [
            InferenceRequest(query="What is the capital of France?", ground_truth="Paris", task_hint="factual", adaptive_depth=True),
            InferenceRequest(query="What element has atomic number 1?", ground_truth="Hydrogen", task_hint="factual", adaptive_depth=True),
            InferenceRequest(query="What year did Apollo 11 land on the Moon?", ground_truth="1969", task_hint="factual", adaptive_depth=True),
        ],
        "arithmetic": [
            InferenceRequest(query="Calculate: 125 * 37", ground_truth="4625", task_hint="arithmetic", tools=True, adaptive_depth=True),
            InferenceRequest(query="What is 4096 / 16 + 25?", ground_truth="281", task_hint="arithmetic", tools=True, adaptive_depth=True),
            InferenceRequest(query="Compute 17^3", ground_truth="4913", task_hint="arithmetic", tools=True, adaptive_depth=True),
        ],
        "medium_reasoning": [
            InferenceRequest(query="If John has 3 apples and gives 1 to Mary, then buys 5 more and gives half of all his apples away, how many does he have?", ground_truth="3.5 or 3", task_hint="reasoning", adaptive_depth=True),
            InferenceRequest(query="A train departs at 2:15 PM and arrives at 4:45 PM. What was the journey duration in hours?", ground_truth="2.5 hours", task_hint="reasoning", adaptive_depth=True),
            InferenceRequest(query="Determine if the sentence 'Every cat loves some fish' logically implies 'Some fish is loved by every cat'.", ground_truth="No, quantifier scope", task_hint="reasoning", adaptive_depth=True),
        ],
        "hard_reasoning": [
            InferenceRequest(query="Analyze the consistency of Zermelo-Fraenkel set theory without the Axiom of Choice and discuss Godel's second incompleteness theorem.", ground_truth="Consistent relative to ZFC", task_hint="reasoning", adaptive_depth=True),
            InferenceRequest(query="Solve: In a village of 100 people, every person who learns a secret tells two new people every hour. How long until all know it?", ground_truth="7 hours", task_hint="reasoning", adaptive_depth=True),
            InferenceRequest(query="Explain why quantum entanglement cannot be used for superluminal communication according to the no-communication theorem.", ground_truth="Reduced density matrix invariant", task_hint="reasoning", adaptive_depth=True),
        ],
        "proof_logic": [
            InferenceRequest(query="Prove that the square root of 2 is irrational using proof by contradiction.", ground_truth="Irrational", task_hint="reasoning", require_verification=True, adaptive_depth=True),
            InferenceRequest(query="Construct a formal truth table proof showing that (P -> Q) is equivalent to (~P v Q).", ground_truth="Equivalent", task_hint="reasoning", require_verification=True, adaptive_depth=True),
            InferenceRequest(query="Prove that every connected graph with V vertices and V-1 edges is a tree.", ground_truth="Tree theorem", task_hint="reasoning", require_verification=True, adaptive_depth=True),
        ],
        "code": [
            InferenceRequest(query="Implement a binary search tree insertion method in Python with balancing.", ground_truth="BST insertion", task_hint="code", adaptive_depth=True),
            InferenceRequest(query="Write a thread-safe singleton pattern in Python using double-checked locking.", ground_truth="Singleton", task_hint="code", adaptive_depth=True),
            InferenceRequest(query="Write an asynchronous token bucket rate limiter in Python with asyncio.", ground_truth="Rate limiter", task_hint="code", adaptive_depth=True),
        ],
        "long_context": [
            InferenceRequest(query="Given the following 500-word corporate policy manual, extract the section covering remote travel reimbursement: " + "Corporate policy paragraph details. " * 30, ground_truth="Travel reimbursement", task_hint="extraction", adaptive_depth=True),
            InferenceRequest(query="Summarize the key architectural patterns discussed across these design documents: " + "Microservices event driven patterns. " * 25, ground_truth="Summary", task_hint="extraction", adaptive_depth=True),
        ],
        "repetitive": [
            InferenceRequest(query="Echo the format: ID: <val>, Status: OK for items 1 through 10 consecutively.", ground_truth="Items 1 to 10", task_hint="extraction", adaptive_depth=True),
            InferenceRequest(query="Repeat the word 'VERIFIED' 15 times separated by commas.", ground_truth="VERIFIED, ...", task_hint="factual", adaptive_depth=True),
        ],
        "uncertainty_heavy": [
            InferenceRequest(query="What is the probability of a catastrophic geomagnetic storm hitting Earth in the next 10 years based on solar cycle 25 models?", ground_truth="Probabilistic range", task_hint="reasoning", adaptive_depth=True),
            InferenceRequest(query="Evaluate whether cold fusion could be physically viable under non-equilibrium lattice conditions.", ground_truth="Uncertain/speculative", task_hint="reasoning", adaptive_depth=True),
        ],
        "verification_heavy": [
            InferenceRequest(query="Verify deterministically if 104729 is a prime number and prove its primality.", ground_truth="104729 is prime", task_hint="reasoning", require_verification=True, tools=True, adaptive_depth=True),
            InferenceRequest(query="Check whether the matrix [[1, 2], [3, 4]] is invertible and compute its exact determinant.", ground_truth="det = -2, invertible", task_hint="symbolic_math", require_verification=True, tools=True, adaptive_depth=True),
        ],
        "mixed": [
            InferenceRequest(query="Calculate 45 * 19 and then explain the prime factors of the resulting product.", ground_truth="855 = 3^2 * 5 * 19", task_hint="arithmetic", tools=True, adaptive_depth=True),
            InferenceRequest(query="Classify sentiment of: 'Buggy update ruined my morning' and write a 2-line Python unit test to reproduce it.", ground_truth="negative + code", task_hint="code", adaptive_depth=True),
        ],
    }


def generate_phase8_hardware_workload() -> Dict[str, List[InferenceRequest]]:
    """
    Generates balanced test workloads for Phase 8 Hardware & Quantization benchmarks.
    Spans memory-heavy, latency-critical, precision-sensitive, and quantization-tolerant tasks.
    """
    return {
        "precision_sensitive": [
            InferenceRequest(query="Prove that the square root of 2 is irrational using proof by contradiction.", ground_truth="Irrational", task_hint="reasoning", require_verification=True, hardware_aware=True),
            InferenceRequest(query="Construct a formal truth table proof showing that (P -> Q) is equivalent to (~P v Q).", ground_truth="Equivalent", task_hint="reasoning", require_verification=True, hardware_aware=True),
            InferenceRequest(query="Analyze the consistency of Zermelo-Fraenkel set theory without the Axiom of Choice and discuss Godel's second incompleteness theorem.", ground_truth="Consistent relative to ZFC", task_hint="reasoning", hardware_aware=True),
        ],
        "quantization_tolerant": [
            InferenceRequest(query="Hi, how are you today?", ground_truth="greeting", task_hint="conversational", hardware_aware=True),
            InferenceRequest(query="Classify sentiment: 'The battery life is exceptional and camera is decent.'", ground_truth="positive", task_hint="classification", hardware_aware=True),
            InferenceRequest(query="Summarize: Artificial Intelligence in medicine is transforming early diagnostics and clinical trial matching.", ground_truth="AI in medicine", task_hint="extraction", hardware_aware=True),
        ],
        "latency_critical": [
            InferenceRequest(query="Quick check: What is the capital of France?", ground_truth="Paris", task_hint="factual", sla_latency_ms=100.0, hardware_aware=True),
            InferenceRequest(query="Classify intent: 'Book an appointment for tomorrow 3pm'", ground_truth="booking", task_hint="classification", sla_latency_ms=120.0, hardware_aware=True),
            InferenceRequest(query="Is 8721 divisible by 3?", ground_truth="Yes", task_hint="arithmetic", sla_latency_ms=150.0, hardware_aware=True),
        ],
        "memory_constrained": [
            InferenceRequest(query="Execute large model reasoning on general relativity and frame dragging around rotating Kerr black holes.", ground_truth="Kerr metric", task_hint="reasoning", memory_budget_mb=4096.0, hardware_aware=True),
            InferenceRequest(query="Solve a multi-variable calculus optimization problem with 4 constraints under tight memory headroom.", ground_truth="Lagrange multipliers", task_hint="reasoning", memory_budget_mb=2048.0, hardware_aware=True),
        ],
        "tool_assisted": [
            InferenceRequest(query="Calculate: 125 * 37 and verify deterministically.", ground_truth="4625", task_hint="arithmetic", tools=True, hardware_aware=True),
            InferenceRequest(query="Compute compound annual growth rate for revenue growing from 100M to 250M over 5 years.", ground_truth="CAGR ~20.11%", task_hint="arithmetic", tools=True, hardware_aware=True),
        ],
    }



