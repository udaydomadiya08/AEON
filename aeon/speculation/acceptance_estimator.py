"""
Online Adaptive Acceptance Rate Estimator for AEON Speculative Decoding.
Conforms to AEON Phase 5 Specification Section 6.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional
from aeon.types import TaskType
from aeon.speculation.types import SpeculationStrategy


class AdaptiveAcceptanceEstimator:
    """
    Learns and tracks P(token accepted | context, strategy, draft_model, draft_length)
    using online Exponential Moving Averages (EMA) and rolling historical statistics.
    Detects acceptance deterioration across decoding rounds to trigger fallback.
    """

    def __init__(self, ema_alpha: float = 0.25, history_size: int = 50):
        self.ema_alpha = ema_alpha
        self.history_size = history_size
        
        # Global and stratified acceptance tracking
        self.global_ema: float = 0.75
        self.strategy_ema: Dict[str, float] = {
            SpeculationStrategy.STANDARD.value: 1.0,
            SpeculationStrategy.DRAFT_MODEL.value: 0.78,
            SpeculationStrategy.NGRAM.value: 0.82,
            SpeculationStrategy.MULTI_TOKEN.value: 0.74,
        }
        self.draft_model_ema: Dict[str, float] = {}
        self.draft_length_ema: Dict[int, float] = {
            1: 0.88,
            2: 0.82,
            4: 0.76,
            6: 0.70,
            8: 0.64,
            16: 0.52,
        }
        
        # Recent round sequence for deterioration detection
        self.round_history: deque = deque(maxlen=history_size)
        self.total_drafted: int = 0
        self.total_accepted: int = 0

    def predict_acceptance(
        self,
        strategy: SpeculationStrategy,
        draft_model: Optional[str] = None,
        draft_length: int = 4,
        task_type: Optional[TaskType] = None,
    ) -> float:
        """
        Estimate the expected token acceptance rate given strategy, draft model,
        draft length, and task classification prior.
        """
        if strategy == SpeculationStrategy.STANDARD:
            return 1.0

        # Base strategy estimate
        strat_score = self.strategy_ema.get(strategy.value, self.global_ema)

        # Draft model modifier
        model_score = self.draft_model_ema.get(draft_model, strat_score) if draft_model else strat_score

        # Draft length attenuation: longer draft prefixes have decaying acceptance
        length_score = self.draft_length_ema.get(draft_length, max(0.40, strat_score * (0.95 ** (draft_length - 1))))

        # Task type prior modulation
        task_prior = 0.0
        if task_type in [TaskType.CODE, TaskType.ARITHMETIC, TaskType.EXTRACTION]:
            task_prior = +0.08  # High syntactic/structural predictability
        elif task_type in [TaskType.REASONING, TaskType.CREATIVE]:
            task_prior = -0.10  # High entropy / branching factor

        # Weighted combination
        combined = (0.35 * strat_score) + (0.35 * model_score) + (0.30 * length_score) + task_prior
        return round(max(0.10, min(0.98, combined)), 4)

    def record_round(
        self,
        strategy: SpeculationStrategy,
        draft_model: Optional[str],
        draft_length: int,
        drafted_count: int,
        accepted_count: int,
    ) -> float:
        """
        Update online EMA and rolling statistics following a completed speculation verification round.
        """
        if drafted_count <= 0:
            return self.global_ema

        round_rate = max(0.0, min(1.0, accepted_count / drafted_count))
        self.total_drafted += drafted_count
        self.total_accepted += accepted_count
        self.round_history.append(round_rate)

        # Update global EMA
        alpha = self.ema_alpha
        self.global_ema = round((alpha * round_rate) + ((1.0 - alpha) * self.global_ema), 4)

        # Update strategy EMA
        strat_key = strategy.value
        prev_strat = self.strategy_ema.get(strat_key, self.global_ema)
        self.strategy_ema[strat_key] = round((alpha * round_rate) + ((1.0 - alpha) * prev_strat), 4)

        # Update draft model EMA
        if draft_model:
            prev_mod = self.draft_model_ema.get(draft_model, self.global_ema)
            self.draft_model_ema[draft_model] = round((alpha * round_rate) + ((1.0 - alpha) * prev_mod), 4)

        # Update draft length EMA
        prev_len = self.draft_length_ema.get(draft_length, self.global_ema)
        self.draft_length_ema[draft_length] = round((alpha * round_rate) + ((1.0 - alpha) * prev_len), 4)

        return round_rate

    def is_deteriorating(self, window: int = 3, delta_threshold: float = 0.15) -> bool:
        """
        Determine if speculative acceptance is consistently collapsing over recent rounds.
        Example: 0.82 -> 0.72 -> 0.61 indicates deterioration, or consecutive collapsed acceptance.
        """
        if len(self.round_history) < window:
            return False

        recent = list(self.round_history)[-window:]
        if all(r < 0.35 for r in recent):
            return True

        is_strictly_decreasing = all(recent[i] < recent[i - 1] for i in range(1, len(recent)))
        total_drop = recent[0] - recent[-1]
        
        return is_strictly_decreasing and (total_drop >= delta_threshold or recent[-1] < 0.40)

    def get_summary(self) -> Dict[str, Any]:
        """Return snapshot of acceptance tracking metrics."""
        return {
            "global_ema": self.global_ema,
            "overall_acceptance_rate": round(self.total_accepted / max(1, self.total_drafted), 4),
            "strategy_ema": dict(self.strategy_ema),
            "draft_length_ema": dict(self.draft_length_ema),
            "rounds_tracked": len(self.round_history),
        }
