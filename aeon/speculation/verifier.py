"""
Speculative Token Verification Engine for AEON.
Conforms to AEON Phase 5 Specification Sections 12, 13, and 30.
"""

from __future__ import annotations

import time
from typing import List, Optional
from aeon.types import ModelProfile
from aeon.speculation.types import (
    DraftTokenProposal,
    VerificationResult,
)


class SpeculationVerifier:
    """
    Evaluates proposed drafted tokens against target-model predictions.
    Determines accepted prefix length, rejection index, and emits replacement tokens.
    Strictly distinguishes SIMULATED vs REAL verification.
    """

    def __init__(self, mode: str = "SIMULATED_SPECULATIVE_DECODING"):
        self.mode = mode

    def verify_draft(
        self,
        target_profile: ModelProfile,
        target_reference_tokens: List[str],
        proposal: DraftTokenProposal,
    ) -> VerificationResult:
        """
        Verify proposed tokens against reference target-model sequence.
        Accepts tokens sequentially until the first divergence.
        """
        t0 = time.perf_counter()

        draft_tokens = proposal.tokens
        accepted: List[str] = []
        rejected: List[str] = []
        replacement_token: Optional[str] = None

        # Compare draft tokens with target model ground-truth / distribution
        for i, tok in enumerate(draft_tokens):
            if i < len(target_reference_tokens) and tok.lower() == target_reference_tokens[i].lower():
                accepted.append(tok)
            else:
                # First point of divergence
                rejected = draft_tokens[i:]
                if i < len(target_reference_tokens):
                    replacement_token = target_reference_tokens[i]
                break

        # If all draft tokens accepted and there is an additional target token available
        if len(accepted) == len(draft_tokens) and len(target_reference_tokens) > len(draft_tokens):
            replacement_token = target_reference_tokens[len(draft_tokens)]

        acceptance_rate = len(accepted) / max(1, len(draft_tokens))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Calibrated verification compute and cost:
        # A batched forward pass over (draft_length + 1) positions takes ~ 1.15x single token latency
        target_token_latency = 1000.0 / max(1.0, target_profile.tokens_per_second)
        verification_latency = target_token_latency * 1.15 + elapsed_ms
        verification_flops = float((len(draft_tokens) + 1) * target_profile.flops_per_token)
        verification_cost = float(((len(accepted) + 1) / 1_000_000.0) * target_profile.cost_per_1m_output_usd)

        return VerificationResult(
            accepted_tokens=accepted,
            rejected_tokens=rejected,
            accepted_length=len(accepted),
            acceptance_rate=round(acceptance_rate, 4),
            replacement_token=replacement_token,
            verification_latency_ms=round(verification_latency, 2),
            verification_cost_usd=round(verification_cost, 7),
            verification_compute_flops=verification_flops,
            target_model=target_profile.name,
            metadata={"execution_mode": self.mode, "divergence_index": len(accepted)},
        )
