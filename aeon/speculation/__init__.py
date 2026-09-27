"""
AEON Speculative and Token-Efficient Inference Package.
Phase 5 — Heterogeneous Computational Allocation for Generation.
"""

from aeon.speculation.types import (
    SpeculationStrategy,
    SpeculationAction,
    DraftTokenProposal,
    VerificationResult,
    BreakEvenAnalysis,
    SpeculationCandidate,
    SpeculationState,
    SpeculationConfig,
)
from aeon.speculation.economics import SpeculationEconomics
from aeon.speculation.acceptance_estimator import AdaptiveAcceptanceEstimator
from aeon.speculation.draft_optimizer import AdaptiveDraftLengthOptimizer
from aeon.speculation.ngram import NgramSpeculationEngine
from aeon.speculation.verifier import SpeculationVerifier
from aeon.speculation.strategies import SpeculativeStrategySelector
from aeon.speculation.controller import (
    SpeculationController,
    global_speculation_controller,
)

__all__ = [
    "SpeculationStrategy",
    "SpeculationAction",
    "DraftTokenProposal",
    "VerificationResult",
    "BreakEvenAnalysis",
    "SpeculationCandidate",
    "SpeculationState",
    "SpeculationConfig",
    "SpeculationEconomics",
    "AdaptiveAcceptanceEstimator",
    "AdaptiveDraftLengthOptimizer",
    "NgramSpeculationEngine",
    "SpeculationVerifier",
    "SpeculativeStrategySelector",
    "SpeculationController",
    "global_speculation_controller",
]
