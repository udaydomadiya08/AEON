"""
Fast Task Classifier for AEON.

Classifies incoming queries into semantic and computational task types
without invoking expensive autoregressive generation.
"""

import re
from typing import Tuple
from aeon.types import TaskType, DifficultyLevel
from aeon.controller.difficulty_estimator import DifficultyEstimator


class TaskClassifier:
    """Fast rule-and-heuristic-based task classifier (Jev-compatible interface)."""

    def __init__(self):
        self.estimator = DifficultyEstimator()

    def classify(self, query: str) -> Tuple[TaskType, float, DifficultyLevel, float]:
        """
        Classifies query into TaskType and calculates difficulty & uncertainty.
        
        Returns:
            (task_type, difficulty_score, difficulty_level, uncertainty_score)
        """
        q_lower = query.lower().strip()

        # Classification / Binary checks
        if any(w in q_lower for w in ["sentiment", "classify", "spam", "is this", "category", "label as", "true or false"]):
            task_type = TaskType.CLASSIFICATION
        # Math & arithmetic
        elif any(w in q_lower for w in ["calculate", "solve", "% of", "equation", "integral", "derivative", "math", "sum of", "product of"]):
            task_type = TaskType.MATH
        # Code & programming
        elif any(w in q_lower for w in ["write a function", "python code", "implement", "debug", "algorithm", "regex", "sql query"]):
            task_type = TaskType.CODE
        # Factual lookups
        elif any(w in q_lower for w in ["what is the capital", "when was", "who wrote", "where is", "capital of", "president of"]):
            task_type = TaskType.FACTUAL
        # Multi-step reasoning
        elif any(w in q_lower for w in ["why does", "explain why", "deduce", "prove that", "reason about", "logic puzzle"]):
            task_type = TaskType.REASONING
        # Creative generation
        elif any(w in q_lower for w in ["write a story", "poem", "essay", "draft an email", "creatively"]):
            task_type = TaskType.CREATIVE
        # Information extraction
        elif any(w in q_lower for w in ["extract", "names from", "entities", "summarize", "key points"]):
            task_type = TaskType.EXTRACTION
        else:
            task_type = TaskType.REASONING

        difficulty, level, uncertainty = self.estimator.estimate(query, task_type)
        return task_type, difficulty, level, uncertainty
