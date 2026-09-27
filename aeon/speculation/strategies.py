"""
Dynamic Speculative Strategy Selector for AEON.
Conforms to AEON Phase 5 Specification Section 8.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from aeon.types import ModelProfile, TaskType
from aeon.models.model_registry import ModelRegistry, global_registry
from aeon.speculation.types import (
    SpeculationStrategy,
    SpeculationCandidate,
    SpeculationConfig,
)
from aeon.speculation.economics import SpeculationEconomics
from aeon.speculation.acceptance_estimator import AdaptiveAcceptanceEstimator
from aeon.speculation.draft_optimizer import AdaptiveDraftLengthOptimizer


class SpeculativeStrategySelector:
    """
    Evaluates candidate speculation strategies (Standard, Draft Model, N-Gram, Multi-Token)
    and selects the strategy maximizing expected economic utility.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        economics: Optional[SpeculationEconomics] = None,
        estimator: Optional[AdaptiveAcceptanceEstimator] = None,
        optimizer: Optional[AdaptiveDraftLengthOptimizer] = None,
        config: Optional[SpeculationConfig] = None,
    ):
        self.registry = registry or global_registry
        self.config = config or SpeculationConfig()
        self.economics = economics or SpeculationEconomics(self.config)
        self.estimator = estimator or AdaptiveAcceptanceEstimator()
        self.optimizer = optimizer or AdaptiveDraftLengthOptimizer(self.economics, self.estimator, self.config)

    def evaluate_all_strategies(
        self,
        target_model: str,
        query: str,
        task_type: Optional[TaskType] = None,
        tokens_to_generate: int = 50,
        preferred_draft_model: Optional[str] = None,
    ) -> List[SpeculationCandidate]:
        """
        Evaluate and rank all supported strategies against the standard decoding baseline.
        """
        target_profile = self.registry.get(target_model)
        if not target_profile:
            # Fallback to standard
            return [
                SpeculationCandidate(
                    strategy=SpeculationStrategy.STANDARD,
                    draft_model=None,
                    draft_length=0,
                    expected_acceptance_rate=1.0,
                    expected_latency_ms=100.0,
                    expected_cost_usd=0.0001,
                    expected_compute_flops=1e9,
                    expected_speedup=1.0,
                    expected_utility=1.0,
                    break_even_acceptance=0.0,
                    is_viable=True,
                    reason="Target model not found in registry; fallback to standard.",
                )
            ]

        candidates: List[SpeculationCandidate] = []

        # 1. Standard Decoding Baseline
        std_candidate = self.economics.evaluate_strategy_candidate(
            strategy=SpeculationStrategy.STANDARD,
            target_profile=target_profile,
            draft_profile=None,
            draft_length=0,
            estimated_acceptance=1.0,
            tokens_to_generate=tokens_to_generate,
        )
        candidates.append(std_candidate)

        if not self.config.enabled:
            return candidates

        # 2. N-Gram Speculation
        ngram_acceptance = self.estimator.predict_acceptance(
            strategy=SpeculationStrategy.NGRAM,
            task_type=task_type,
        )
        opt_len_ngram, _, _ = self.optimizer.optimize_draft_length(
            strategy=SpeculationStrategy.NGRAM,
            target_profile=target_profile,
            draft_profile=None,
            current_acceptance=ngram_acceptance,
        )
        ngram_candidate = self.economics.evaluate_strategy_candidate(
            strategy=SpeculationStrategy.NGRAM,
            target_profile=target_profile,
            draft_profile=None,
            draft_length=opt_len_ngram,
            estimated_acceptance=ngram_acceptance,
            tokens_to_generate=tokens_to_generate,
        )
        candidates.append(ngram_candidate)

        # 3. Draft Model Speculation
        compatible_drafts = self.registry.list_draft_models(target_model)
        selected_draft = None
        if preferred_draft_model and self.registry.is_compatible(preferred_draft_model, target_model):
            selected_draft = self.registry.get(preferred_draft_model)
        elif compatible_drafts:
            selected_draft = self.registry.get_best_draft_model(target_model)

        if selected_draft:
            draft_acceptance = self.estimator.predict_acceptance(
                strategy=SpeculationStrategy.DRAFT_MODEL,
                draft_model=selected_draft.name,
                task_type=task_type,
            )
            opt_len_draft, _, _ = self.optimizer.optimize_draft_length(
                strategy=SpeculationStrategy.DRAFT_MODEL,
                target_profile=target_profile,
                draft_profile=selected_draft,
                current_acceptance=draft_acceptance,
            )
            draft_candidate = self.economics.evaluate_strategy_candidate(
                strategy=SpeculationStrategy.DRAFT_MODEL,
                target_profile=target_profile,
                draft_profile=selected_draft,
                draft_length=opt_len_draft,
                estimated_acceptance=draft_acceptance,
                tokens_to_generate=tokens_to_generate,
            )
            candidates.append(draft_candidate)

        # 4. Multi-Token Speculation
        mt_acceptance = self.estimator.predict_acceptance(
            strategy=SpeculationStrategy.MULTI_TOKEN,
            task_type=task_type,
        )
        mt_candidate = self.economics.evaluate_strategy_candidate(
            strategy=SpeculationStrategy.MULTI_TOKEN,
            target_profile=target_profile,
            draft_profile=None,
            draft_length=2,
            estimated_acceptance=mt_acceptance,
            tokens_to_generate=tokens_to_generate,
        )
        candidates.append(mt_candidate)

        # Sort by expected utility descending
        candidates.sort(key=lambda c: (c.is_viable, c.expected_utility), reverse=True)
        return candidates

    def select_strategy(
        self,
        target_model: str,
        query: str,
        task_type: Optional[TaskType] = None,
        tokens_to_generate: int = 50,
        preferred_draft_model: Optional[str] = None,
    ) -> Tuple[SpeculationCandidate, List[SpeculationCandidate]]:
        """
        Select the optimal speculation strategy candidate.
        If no speculative candidate is viable or beats standard decoding, selects STANDARD.
        """
        candidates = self.evaluate_all_strategies(
            target_model=target_model,
            query=query,
            task_type=task_type,
            tokens_to_generate=tokens_to_generate,
            preferred_draft_model=preferred_draft_model,
        )
        top = candidates[0]
        # If top candidate is not viable or has negative marginal utility vs standard, revert to standard
        if not top.is_viable or (top.strategy != SpeculationStrategy.STANDARD and top.marginal_utility <= 0):
            std = next((c for c in candidates if c.strategy == SpeculationStrategy.STANDARD), top)
            return std, candidates

        return top, candidates
