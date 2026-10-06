"""
AEON Phase 7 — Token Importance Estimator.

Estimates token/position importance for Mixture-of-Depths (MoD) routing.
Distinguishes high-value content tokens requiring deep reasoning layers
from low-entropy structural tokens that can safely exit or skip layers.
"""

import re
from typing import List, Tuple, Any, Optional
from aeon.depth.types import TokenComputeState, TokenImportanceSignal


# Common English structural / low-entropy tokens
LOW_ENTROPY_TOKENS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "it", "its", "that", "this", "these", "those",
    ",", ".", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "\"", "'",
}

HIGH_ENTROPY_INDICATORS = {
    "prove", "calculate", "solve", "why", "how", "what", "integral", "derivative",
    "bipartite", "halting", "turing", "theorem", "prime", "algorithm", "optimal",
    "if", "then", "therefore", "contradiction", "complexity", "hypothesis",
}


class TokenImportanceEstimator:
    """
    Estimates computational importance for token sequences.
    Labels mechanism as SIMULATED_TOKEN_IMPORTANCE when no raw attention tensors exist.
    """

    def __init__(self, mode: TokenImportanceSignal = TokenImportanceSignal.SEMANTIC_IMPORTANCE):
        self.mode = mode

    def tokenize_and_estimate(
        self,
        text: str,
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
        min_layers: int = 4,
        max_layers: int = 32,
    ) -> List[TokenComputeState]:
        """
        Tokenize input text and compute per-token importance scores in [0.1, 1.0].
        Returns a list of TokenComputeState ready for MoD routing.
        """
        if not text.strip():
            return [
                TokenComputeState(
                    token_id=0,
                    token_text="<empty>",
                    importance=0.5,
                    min_layers=min_layers,
                    max_layers=max_layers,
                    allocated_layers=min_layers,
                )
            ]

        # Simple word & punctuation tokenization
        raw_tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        if not raw_tokens:
            raw_tokens = [text]

        token_states: List[TokenComputeState] = []
        n_tokens = len(raw_tokens)

        for idx, tok in enumerate(raw_tokens):
            tok_lower = tok.lower()
            importance = self._score_token(tok_lower, idx, n_tokens, difficulty, uncertainty)

            token_states.append(
                TokenComputeState(
                    token_id=idx,
                    token_text=tok,
                    importance=round(importance, 4),
                    min_layers=min_layers,
                    max_layers=max_layers,
                    allocated_layers=min_layers,
                )
            )

        return token_states

    def _score_token(
        self,
        tok: str,
        pos: int,
        total: int,
        difficulty: float,
        uncertainty: float,
    ) -> float:
        """Score an individual token's computational value."""
        if self.mode == TokenImportanceSignal.UNIFORM:
            return 0.5

        # Base importance
        if tok in HIGH_ENTROPY_INDICATORS:
            base = 0.95
        elif tok.isnumeric():
            base = 0.90
        elif tok in LOW_ENTROPY_TOKENS:
            base = 0.20
        elif len(tok) > 6:
            base = 0.75
        elif len(tok) <= 2 and not tok.isalnum():
            base = 0.15
        else:
            base = 0.55

        # Position weight: first tokens (query intent) and terminal tokens (conclusions) are slightly more critical
        pos_ratio = pos / float(max(1, total - 1))
        pos_bonus = 0.10 if (pos_ratio < 0.2 or pos_ratio > 0.8) else 0.0

        # Difficulty & uncertainty interaction
        diff_factor = max(0.0, min(1.0, difficulty)) * 0.15
        unc_factor = max(0.0, min(1.0, uncertainty)) * 0.10

        score = base + pos_bonus + diff_factor + unc_factor
        return max(0.10, min(1.0, score))

    def estimate_importance(
        self,
        tokens: List[str],
        query: str = "",
        task_type: Any = None,
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
    ) -> List[float]:
        """Compute raw normalized importance scores in [0.0, 1.0] for list of tokens."""
        if not tokens:
            return []
        total = len(tokens)
        return [
            self._score_token(
                tok=t,
                pos=i,
                total=total,
                difficulty=difficulty,
                uncertainty=uncertainty,
            )
            for i, t in enumerate(tokens)
        ]


global_token_importance_estimator = TokenImportanceEstimator()
