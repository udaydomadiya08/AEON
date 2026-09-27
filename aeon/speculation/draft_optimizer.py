"""
Adaptive Draft Length Optimizer for Speculative Decoding.
Conforms to AEON Phase 5 Specification Section 7.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from aeon.types import ModelProfile
from aeon.speculation.types import SpeculationStrategy, SpeculationConfig
from aeon.speculation.economics import SpeculationEconomics
from aeon.speculation.acceptance_estimator import AdaptiveAcceptanceEstimator


class AdaptiveDraftLengthOptimizer:
    """
    Evaluates Candidate Draft Lengths (e.g., 1, 2, 4, 6, 8, 16)
    and dynamically selects the length maximizing expected utility and speedup.
    """

    def __init__(
        self,
        economics: Optional[SpeculationEconomics] = None,
        estimator: Optional[AdaptiveAcceptanceEstimator] = None,
        config: Optional[SpeculationConfig] = None,
    ):
        self.config = config or SpeculationConfig()
        self.economics = economics or SpeculationEconomics(self.config)
        self.estimator = estimator or AdaptiveAcceptanceEstimator()
        self.candidate_lengths: List[int] = self.config.candidate_draft_lengths

    def optimize_draft_length(
        self,
        strategy: SpeculationStrategy,
        target_profile: ModelProfile,
        draft_profile: Optional[ModelProfile],
        current_acceptance: Optional[float] = None,
    ) -> Tuple[int, float, str]:
        """
        Evaluate candidate draft lengths and return:
        (optimal_length, expected_speedup, explanation)
        """
        if strategy == SpeculationStrategy.STANDARD:
            return 0, 1.0, "Standard decoding uses no draft length."

        best_length = 4
        best_utility = -float("inf")
        best_speedup = 1.0
        details = []

        base_acceptance = current_acceptance if current_acceptance is not None else self.estimator.global_ema

        for length in self.candidate_lengths:
            # Model acceptance degradation over sequence length: longer sequences have lower cumulative acceptance
            # alpha(k) ~ alpha * (0.97 ^ (k - 1))
            length_decay = 0.97 ** (length - 1)
            effective_acceptance = max(0.20, base_acceptance * length_decay)

            candidate = self.economics.evaluate_strategy_candidate(
                strategy=strategy,
                target_profile=target_profile,
                draft_profile=draft_profile,
                draft_length=length,
                estimated_acceptance=effective_acceptance,
            )

            details.append((length, candidate.expected_speedup, candidate.expected_utility))

            if candidate.expected_utility > best_utility:
                best_utility = candidate.expected_utility
                best_length = length
                best_speedup = candidate.expected_speedup

        explanation = (
            f"Selected optimal draft length {best_length} (expected speedup {best_speedup}x, "
            f"utility {best_utility:.4f}) over candidates {self.candidate_lengths}."
        )

        return best_length, best_speedup, explanation

    def adapt_length_on_feedback(
        self,
        current_length: int,
        realized_acceptance: float,
        break_even_rate: float,
    ) -> Tuple[int, str]:
        """
        Adjust draft length dynamically between rounds based on empirical acceptance rate.
        If acceptance is high, expand draft length; if degrading, contract draft length.
        """
        lengths = sorted(self.candidate_lengths)
        idx = lengths.index(current_length) if current_length in lengths else 2

        if realized_acceptance < break_even_rate or realized_acceptance < 0.45:
            # Step down to reduce waste
            new_idx = max(0, idx - 1)
            new_length = lengths[new_idx]
            reason = f"Acceptance ({realized_acceptance:.1%}) below break-even ({break_even_rate:.1%}); reducing draft length {current_length} -> {new_length}."
            return new_length, reason
        elif realized_acceptance > 0.85 and idx < len(lengths) - 1:
            # Step up to exploit high predictability
            new_idx = idx + 1
            new_length = lengths[new_idx]
            reason = f"High acceptance ({realized_acceptance:.1%}); increasing draft length {current_length} -> {new_length}."
            return new_length, reason
        else:
            return current_length, f"Acceptance ({realized_acceptance:.1%}) stable; maintaining draft length {current_length}."
