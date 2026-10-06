"""
Difficulty and Uncertainty Estimator for AEON.

Evaluates the intrinsic computational complexity and cognitive demand of a query
to inform the compute budget controller and routing decisions.
"""

import re
from typing import Tuple
from aeon.types import DifficultyLevel, TaskType


class DifficultyEstimator:
    """Estimates difficulty score [0.0, 1.0] and uncertainty [0.0, 1.0] for incoming queries."""

    # Patterns indicating high computational or reasoning difficulty
    MATH_SYMBOLS = re.compile(r"[\+\-\*\/\^=<>∑∏∫√%≤≥≠]")
    LOGICAL_MARKERS = re.compile(r"\b(prove|proof|deduce|if and only if|contradiction|contrapositive|axiom|hypothesis|conclude|therefore|theorem|lemma)\b", re.IGNORECASE)
    CODE_MARKERS = re.compile(r"\b(def|class|function|return|import|async|await|pointer|recursion|algorithm|time complexity)\b", re.IGNORECASE)
    MULTI_STEP_MARKERS = re.compile(r"\b(step-by-step|first.*then|subsequently|compare and contrast|evaluate.*tradeoffs|multi-step)\b", re.IGNORECASE)
    AMBIGUITY_MARKERS = re.compile(r"\b(maybe|perhaps|opinion|subjective|speculate|could|might|unclear)\b", re.IGNORECASE)

    def estimate(self, query: str, task_type: TaskType) -> Tuple[float, DifficultyLevel, float]:
        """
        Returns:
            difficulty_score: float in [0.0, 1.0]
            difficulty_level: DifficultyLevel (EASY, MEDIUM, HARD)
            uncertainty_score: float in [0.0, 1.0]
        """
        score = 0.20  # Base difficulty

        # Length factor
        words = len(query.strip().split())
        if words > 120:
            score += 0.25
        elif words > 50:
            score += 0.15
        elif words < 12:
            score -= 0.05

        # Task type baseline adjustment
        if task_type in [TaskType.CLASSIFICATION, TaskType.ARITHMETIC]:
            score -= 0.15
        elif task_type in [TaskType.FACTUAL, TaskType.RETRIEVAL]:
            score -= 0.10
        elif task_type in [TaskType.MATH, TaskType.SYMBOLIC_MATH]:
            score += 0.25
        elif task_type in [TaskType.CODE, TaskType.FINANCIAL, TaskType.DATA_ANALYSIS]:
            score += 0.20
        elif task_type == TaskType.REASONING:
            score += 0.20

        # Feature detectors
        if self.MATH_SYMBOLS.search(query):
            score += 0.15
        logical_matches = len(self.LOGICAL_MARKERS.findall(query))
        if logical_matches > 0:
            score += min(0.35, 0.15 * logical_matches)
        if self.CODE_MARKERS.search(query):
            score += 0.15
        if self.MULTI_STEP_MARKERS.search(query):
            score += 0.20

        difficulty = max(0.05, min(0.98, round(score, 3)))

        # Categorize level
        if difficulty < 0.40:
            level = DifficultyLevel.EASY
        elif difficulty < 0.70:
            level = DifficultyLevel.MEDIUM
        else:
            level = DifficultyLevel.HARD

        # Uncertainty estimation
        ambiguity_matches = len(self.AMBIGUITY_MARKERS.findall(query))
        uncertainty = max(0.05, min(0.95, round(0.10 + 0.15 * ambiguity_matches + 0.20 * (1.0 - abs(difficulty - 0.5) * 2), 3)))

        return difficulty, level, uncertainty
