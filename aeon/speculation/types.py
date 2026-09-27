"""
Type definitions and data models for AEON Speculative and Token-Efficient Inference.
Conforms to AEON Phase 5 Specification Sections 4, 5, 12, 19, 24, and 32.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SpeculationStrategy(str, Enum):
    """Supported speculative decoding strategies in AEON."""
    NONE = "none"
    STANDARD = "standard"
    DRAFT_MODEL = "draft_model"
    NGRAM = "ngram"
    MULTI_TOKEN = "multi_token"
    HYBRID = "hybrid"


class SpeculationAction(str, Enum):
    """Discrete control actions within the speculative execution loop."""
    START_SPECULATION = "START_SPECULATION"
    CONTINUE_SPECULATION = "CONTINUE_SPECULATION"
    REDUCE_DRAFT_LENGTH = "REDUCE_DRAFT_LENGTH"
    INCREASE_DRAFT_LENGTH = "INCREASE_DRAFT_LENGTH"
    CHANGE_DRAFT_MODEL = "CHANGE_DRAFT_MODEL"
    DISABLE_SPECULATION = "DISABLE_SPECULATION"
    FALLBACK_STANDARD = "FALLBACK_STANDARD"
    VERIFY = "VERIFY"
    STOP = "STOP"


class DraftTokenProposal(BaseModel):
    """Proposed candidate continuation tokens produced by a draft mechanism."""
    tokens: List[str]
    confidences: List[float] = Field(default_factory=list)
    strategy: SpeculationStrategy
    draft_model: Optional[str] = None
    draft_length: int
    draft_latency_ms: float = 0.0
    draft_cost_usd: float = 0.0
    draft_compute_flops: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of validating drafted tokens against the target model distribution."""
    accepted_tokens: List[str]
    rejected_tokens: List[str] = Field(default_factory=list)
    accepted_length: int
    acceptance_rate: float
    replacement_token: Optional[str] = None
    verification_latency_ms: float = 0.0
    verification_cost_usd: float = 0.0
    verification_compute_flops: float = 0.0
    target_model: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BreakEvenAnalysis(BaseModel):
    """
    Economic break-even evaluation for speculative inference.
    Determines the minimum token acceptance rate and threshold conditions
    under which speculation outperforms standard autoregressive decoding.
    """
    min_acceptance_rate: float
    expected_speedup: float
    cost_parity_acceptance: float
    latency_parity_acceptance: float
    is_favorable: bool
    explanation: str


class SpeculationCandidate(BaseModel):
    """Evaluated speculation candidate strategy with projected economics and VOI."""
    strategy: SpeculationStrategy
    draft_model: Optional[str] = None
    draft_length: int = 4
    expected_acceptance_rate: float
    expected_latency_ms: float
    expected_cost_usd: float
    expected_compute_flops: float
    expected_speedup: float
    expected_utility: float
    marginal_utility: float = 0.0
    voi_ratio: float = 0.0
    break_even_acceptance: float
    is_viable: bool
    reason: str


class SpeculationState(BaseModel):
    """
    Explicit runtime state tracked across speculative decoding rounds.
    Conforms to Phase 5 Specification Section 4.
    """
    request_id: str
    target_model: str
    current_strategy: SpeculationStrategy = SpeculationStrategy.STANDARD
    draft_model: Optional[str] = None
    draft_length: int = 4
    
    # Token counters
    generated_tokens: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    rounds_executed: int = 0
    
    # Acceptance stats
    acceptance_rate: float = 0.0
    average_accepted_tokens_per_round: float = 0.0
    rolling_acceptance_history: List[float] = Field(default_factory=list)
    
    # Performance & Economics
    draft_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    draft_compute_flops: float = 0.0
    verification_compute_flops: float = 0.0
    total_compute_flops: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_energy_joules: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Dynamic decision metrics
    confidence: float = 1.0
    expected_speedup: float = 1.0
    realized_speedup: float = 1.0
    expected_utility: float = 0.0
    realized_utility: float = 0.0
    
    # Execution labels
    execution_mode: str = "SIMULATED_SPECULATIVE_DECODING"
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)


class SpeculationConfig(BaseModel):
    """Configuration parameters for AEON Speculation Controller."""
    enabled: bool = True
    strategies: List[SpeculationStrategy] = Field(
        default_factory=lambda: [
            SpeculationStrategy.STANDARD,
            SpeculationStrategy.DRAFT_MODEL,
            SpeculationStrategy.NGRAM,
            SpeculationStrategy.MULTI_TOKEN,
        ]
    )
    candidate_draft_lengths: List[int] = Field(default_factory=lambda: [1, 2, 4, 6, 8, 16])
    default_draft_length: int = 4
    min_acceptance_rate: float = 0.50
    fallback_threshold: float = 0.35
    voi_threshold: float = 0.0
    learning_rate: float = 0.25
    max_speculation_rounds: int = 50
    latency_weight: float = 0.00001
    cost_weight: float = 2.0
    compute_weight: float = 1e-15
    memory_weight: float = 0.000001
    energy_weight: float = 0.00002
    risk_weight: float = 0.05
