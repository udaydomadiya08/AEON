"""
N-Gram Prompt and Context Speculation Engine for AEON.
Conforms to AEON Phase 5 Specification Section 11.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from aeon.speculation.types import (
    DraftTokenProposal,
    SpeculationStrategy,
)


class NgramSpeculationEngine:
    """
    Lightweight, deterministic n-gram continuation engine.
    Finds repeated token n-grams in the prompt and recent generated tokens to speculate
    future continuations without GPU compute overhead.
    """

    def __init__(self, min_order: int = 2, max_order: int = 5):
        self.min_order = min_order
        self.max_order = max_order
        # Cache of n-gram transitions: order -> prefix_tuple -> Dict[next_token, count]
        self._table: Dict[int, Dict[Tuple[str, ...], Dict[str, int]]] = {
            k: defaultdict(lambda: defaultdict(int)) for k in range(min_order, max_order + 1)
        }

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace/punctuation tokenization preserving subwords."""
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def index_context(self, context_text: str) -> None:
        """Populate the n-gram transition table from input prompt or context."""
        tokens = self.tokenize(context_text)
        if len(tokens) < self.min_order + 1:
            return

        for order in range(self.min_order, min(self.max_order + 1, len(tokens))):
            table = self._table[order]
            for i in range(len(tokens) - order):
                prefix = tuple(tokens[i : i + order])
                next_tok = tokens[i + order]
                table[prefix][next_tok] += 1

    def speculate(
        self,
        current_sequence: str,
        draft_length: int = 4,
    ) -> Optional[DraftTokenProposal]:
        """
        Predict up to `draft_length` continuation tokens using the longest matching n-gram prefix.
        Returns None if no matching sequence is found.
        """
        t0 = time.perf_counter()
        tokens = self.tokenize(current_sequence)
        if len(tokens) < self.min_order:
            return None

        # Build local context model on current sequence
        self.index_context(current_sequence)

        drafted: List[str] = []
        confidences: List[float] = []
        seq_tokens = list(tokens)

        for _ in range(draft_length):
            found_next = False
            # Check orders from highest (max_order) down to min_order
            for order in range(min(self.max_order, len(seq_tokens)), self.min_order - 1, -1):
                prefix = tuple(seq_tokens[-order:])
                candidates = self._table[order].get(prefix)
                if candidates:
                    total_counts = sum(candidates.values())
                    best_tok, best_count = max(candidates.items(), key=lambda kv: kv[1])
                    conf = best_count / total_counts
                    if conf >= 0.50:
                        drafted.append(best_tok)
                        confidences.append(round(conf, 3))
                        seq_tokens.append(best_tok)
                        found_next = True
                        break

            if not found_next:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not drafted:
            return None

        return DraftTokenProposal(
            tokens=drafted,
            confidences=confidences,
            strategy=SpeculationStrategy.NGRAM,
            draft_model="ngram-table",
            draft_length=len(drafted),
            draft_latency_ms=round(elapsed_ms, 3),
            draft_cost_usd=0.0,
            draft_compute_flops=float(len(drafted) * 5000),
            metadata={"ngram_orders_checked": list(range(self.min_order, self.max_order + 1))},
        )

    def clear(self) -> None:
        """Reset internal n-gram tables."""
        for order in self._table:
            self._table[order].clear()
