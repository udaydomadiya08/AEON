"""
Economic Model and Break-Even Analyzer for Speculative Inference.
Conforms to AEON Phase 5 Specification Sections 5, 10, 16, and 24.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
from aeon.types import ModelProfile, UtilityWeights
from aeon.speculation.types import (
    SpeculationStrategy,
    SpeculationCandidate,
    BreakEvenAnalysis,
    SpeculationConfig,
)


class SpeculationEconomics:
    """
    Evaluates formal economic utility, marginal VOI, and break-even acceptance rates
    for speculative decoding vs standard autoregressive generation.
    """

    def __init__(self, config: Optional[SpeculationConfig] = None):
        self.config = config or SpeculationConfig()

    def compute_utility(
        self,
        quality: float,
        latency_ms: float,
        cost_usd: float,
        compute_flops: float,
        memory_mb: float = 0.0,
        energy_joules: float = 0.0,
        risk: float = 0.0,
    ) -> float:
        """
        Formal AEON Utility calculation:
        U = Q - λ_lat * L - λ_cost * C - λ_compute * F - λ_mem * M - λ_energy * E - λ_risk * R
        """
        cfg = self.config
        return (
            quality
            - (cfg.latency_weight * latency_ms)
            - (cfg.cost_weight * cost_usd)
            - (cfg.compute_weight * compute_flops)
            - (cfg.memory_weight * memory_mb)
            - (cfg.energy_weight * energy_joules)
            - (cfg.risk_weight * risk)
        )

    def calculate_expected_accepted_tokens(
        self,
        draft_length: int,
        acceptance_rate: float,
    ) -> float:
        """
        Calculate expected accepted tokens per speculation step.
        Each drafted token has conditional acceptance probability ~ α^i.
        Including the bonus/replacement token verified by the target model,
        expected tokens emitted per round = 1 + sum_{i=1}^{gamma} alpha^i.
        """
        alpha = max(0.01, min(0.999, acceptance_rate))
        if abs(alpha - 1.0) < 1e-4:
            return float(draft_length + 1)
        # Sum of geometric progression for drafted tokens
        expected_draft_accepted = (alpha * (1.0 - math.pow(alpha, draft_length))) / (1.0 - alpha)
        # Add 1 verified/replacement target token emitted in the same forward pass
        return 1.0 + expected_draft_accepted

    def calculate_break_even(
        self,
        target_profile: ModelProfile,
        draft_profile: Optional[ModelProfile],
        draft_length: int,
        strategy: SpeculationStrategy = SpeculationStrategy.DRAFT_MODEL,
    ) -> BreakEvenAnalysis:
        """
        Compute theoretical break-even acceptance rate where speculative execution
        matches or exceeds the latency and cost of standard decoding.
        """
        target_time_per_token = 1000.0 / max(1.0, target_profile.tokens_per_second)
        target_verify_overhead = target_time_per_token * 1.15  # Batch verification forward pass

        if strategy == SpeculationStrategy.NGRAM:
            # N-gram drafting has zero GPU cost and ~0.05ms CPU latency
            draft_time_total = 0.05
            draft_cost_total = 0.0
        elif draft_profile:
            draft_time_per_token = 1000.0 / max(1.0, draft_profile.tokens_per_second)
            draft_time_total = draft_length * draft_time_per_token
            draft_cost_total = (draft_length / 1_000_000.0) * draft_profile.cost_per_1m_output_usd
        else:
            return BreakEvenAnalysis(
                min_acceptance_rate=1.0,
                expected_speedup=1.0,
                cost_parity_acceptance=1.0,
                latency_parity_acceptance=1.0,
                is_favorable=False,
                explanation="No draft profile available for break-even calculation.",
            )

        # Time for one round of speculative decoding
        time_per_round = draft_time_total + target_verify_overhead

        # For speedup S > 1.0: (Tokens per round * target_time_per_token) > time_per_round
        # Tokens_per_round > time_per_round / target_time_per_token
        req_tokens_per_round = time_per_round / target_time_per_token
        
        # Binary search for acceptance rate alpha where expected_tokens >= req_tokens_per_round
        low, high = 0.05, 0.99
        min_alpha = high
        for _ in range(25):
            mid = (low + high) / 2.0
            tokens = self.calculate_expected_accepted_tokens(draft_length, mid)
            if tokens >= req_tokens_per_round:
                min_alpha = mid
                high = mid
            else:
                low = mid

        # Cost parity: speculative decoding incurs draft model tokens in addition to target tokens
        target_token_cost = target_profile.cost_per_1m_output_usd / 1_000_000.0
        # Cost is favorable if drafted tokens yield enough speedup without exceeding cost threshold
        cost_parity = min(0.95, min_alpha * (1.0 + (draft_cost_total / max(1e-9, target_token_cost))))

        # Expected speedup at typical acceptance rate (0.75)
        sample_tokens = self.calculate_expected_accepted_tokens(draft_length, 0.75)
        standard_time = sample_tokens * target_time_per_token
        expected_speedup = round(standard_time / max(0.01, time_per_round), 2)

        is_favorable = min_alpha < 0.85
        explanation = (
            f"Break-even acceptance rate is {min_alpha:.1%}. "
            f"At 75% acceptance, expected latency speedup is {expected_speedup}x."
        )

        return BreakEvenAnalysis(
            min_acceptance_rate=round(min_alpha, 3),
            expected_speedup=expected_speedup,
            cost_parity_acceptance=round(cost_parity, 3),
            latency_parity_acceptance=round(min_alpha, 3),
            is_favorable=is_favorable,
            explanation=explanation,
        )

    def evaluate_strategy_candidate(
        self,
        strategy: SpeculationStrategy,
        target_profile: ModelProfile,
        draft_profile: Optional[ModelProfile],
        draft_length: int,
        estimated_acceptance: float,
        tokens_to_generate: int = 50,
    ) -> SpeculationCandidate:
        """
        Evaluate full economic metrics, speedup, and utility for a candidate speculation strategy.
        """
        target_time_per_token = 1000.0 / max(1.0, target_profile.tokens_per_second)
        target_cost_per_token = target_profile.cost_per_1m_output_usd / 1_000_000.0
        target_flops_per_token = target_profile.flops_per_token

        # Baseline Standard Decoding
        std_latency_ms = target_profile.base_latency_ms + (tokens_to_generate * target_time_per_token)
        std_cost_usd = tokens_to_generate * target_cost_per_token
        std_compute_flops = tokens_to_generate * target_flops_per_token
        std_utility = self.compute_utility(
            quality=target_profile.quality_score,
            latency_ms=std_latency_ms,
            cost_usd=std_cost_usd,
            compute_flops=std_compute_flops,
            memory_mb=target_profile.min_memory_mb,
        )

        if strategy == SpeculationStrategy.STANDARD:
            return SpeculationCandidate(
                strategy=SpeculationStrategy.STANDARD,
                draft_model=None,
                draft_length=0,
                expected_acceptance_rate=1.0,
                expected_latency_ms=round(std_latency_ms, 2),
                expected_cost_usd=round(std_cost_usd, 7),
                expected_compute_flops=std_compute_flops,
                expected_speedup=1.0,
                expected_utility=round(std_utility, 4),
                marginal_utility=0.0,
                voi_ratio=0.0,
                break_even_acceptance=0.0,
                is_viable=True,
                reason="Standard autoregressive baseline.",
            )

        # Break-Even Analysis
        break_even = self.calculate_break_even(
            target_profile=target_profile,
            draft_profile=draft_profile,
            draft_length=draft_length,
            strategy=strategy,
        )

        # Tokens accepted per round
        tokens_per_round = self.calculate_expected_accepted_tokens(draft_length, estimated_acceptance)
        rounds_needed = math.ceil(tokens_to_generate / max(1.0, tokens_per_round))

        # Drafting costs per round
        if strategy == SpeculationStrategy.NGRAM:
            draft_latency_round = 0.08
            draft_cost_round = 0.0
            draft_flops_round = 5e4
            draft_mem = 2.0
            draft_name = "ngram-cache"
        elif strategy == SpeculationStrategy.MULTI_TOKEN:
            # Multi-token head prediction: single forward pass with multi-head output
            draft_latency_round = target_time_per_token * 0.25
            draft_cost_round = target_cost_per_token * 0.10 * draft_length
            draft_flops_round = target_flops_per_token * 0.15 * draft_length
            draft_mem = target_profile.min_memory_mb * 0.05
            draft_name = "multi-token-head"
        elif draft_profile:
            draft_time_per_token = 1000.0 / max(1.0, draft_profile.tokens_per_second)
            draft_latency_round = draft_length * draft_time_per_token
            draft_cost_round = (draft_length / 1_000_000.0) * draft_profile.cost_per_1m_output_usd
            draft_flops_round = draft_length * draft_profile.flops_per_token
            draft_mem = draft_profile.min_memory_mb
            draft_name = draft_profile.name
        else:
            return SpeculationCandidate(
                strategy=strategy,
                draft_model=None,
                draft_length=draft_length,
                expected_acceptance_rate=0.0,
                expected_latency_ms=std_latency_ms,
                expected_cost_usd=std_cost_usd,
                expected_compute_flops=std_compute_flops,
                expected_speedup=0.5,
                expected_utility=-999.0,
                break_even_acceptance=1.0,
                is_viable=False,
                reason="Missing draft profile.",
            )

        # Verification cost per round (parallel forward pass over draft sequence)
        verify_latency_round = target_time_per_token * 1.15
        verify_cost_round = (tokens_per_round / 1_000_000.0) * target_profile.cost_per_1m_output_usd
        verify_flops_round = (draft_length + 1) * target_flops_per_token

        # Totals across all rounds
        total_latency_ms = target_profile.base_latency_ms + (rounds_needed * (draft_latency_round + verify_latency_round))
        total_cost_usd = rounds_needed * (draft_cost_round + verify_cost_round)
        total_compute_flops = rounds_needed * (draft_flops_round + verify_flops_round)
        total_mem = target_profile.min_memory_mb + draft_mem

        # Speedup relative to standard decoding
        speedup = round(std_latency_ms / max(0.01, total_latency_ms), 2)

        # Speculative utility
        spec_utility = self.compute_utility(
            quality=target_profile.quality_score,
            latency_ms=total_latency_ms,
            cost_usd=total_cost_usd,
            compute_flops=total_compute_flops,
            memory_mb=total_mem,
            risk=0.01 if estimated_acceptance >= break_even.min_acceptance_rate else 0.15,
        )

        marginal_u = spec_utility - std_utility
        marginal_cost = max(1e-7, total_cost_usd - std_cost_usd if total_cost_usd > std_cost_usd else 1e-6)
        voi_ratio = marginal_u / marginal_cost if marginal_u > 0 else 0.0

        is_viable = (estimated_acceptance >= break_even.min_acceptance_rate) and (speedup >= 1.05 or marginal_u > 0)

        reason = (
            f"Expected speedup {speedup}x (acceptance {estimated_acceptance:.1%}, break-even {break_even.min_acceptance_rate:.1%}). "
            f"Utility delta: {marginal_u:+.4f} vs standard."
        )

        return SpeculationCandidate(
            strategy=strategy,
            draft_model=draft_name,
            draft_length=draft_length,
            expected_acceptance_rate=round(estimated_acceptance, 3),
            expected_latency_ms=round(total_latency_ms, 2),
            expected_cost_usd=round(total_cost_usd, 7),
            expected_compute_flops=total_compute_flops,
            expected_speedup=speedup,
            expected_utility=round(spec_utility, 4),
            marginal_utility=round(marginal_u, 4),
            voi_ratio=round(voi_ratio, 2),
            break_even_acceptance=break_even.min_acceptance_rate,
            is_viable=is_viable,
            reason=reason,
        )
