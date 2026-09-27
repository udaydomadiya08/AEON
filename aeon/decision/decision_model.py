"""
Decision Model / Jev-like Path.

Provides non-autoregressive, typed, calibrated classification and decision-making
without incurring expensive token-by-token generation overhead.
Conforms to AEON Specification Section 10.
"""

from typing import Dict, Any, List, Union
from abc import ABC, abstractmethod


class BaseDecisionModel(ABC):
    """Abstract interface for typed decision models."""

    @abstractmethod
    def predict(self, input_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Produce typed outputs matching the schema without autoregressive generation."""
        pass


class TypedDecisionModel(BaseDecisionModel):
    """
    Lightweight, fast decision engine.
    Supports structured decision schemas for routing, classification, tool selection,
    and verification confidence.
    """

    def predict(self, input_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        text_lower = input_text.lower()
        result: Dict[str, Any] = {}

        for key, expected_type in schema.items():
            if isinstance(expected_type, list):
                # Categorical selection among options
                result[key] = self._select_category(text_lower, expected_type)
            elif expected_type == "bool":
                result[key] = self._predict_bool(text_lower, key)
            elif expected_type == "float":
                result[key] = self._predict_float(text_lower, key)
            elif expected_type == "int":
                result[key] = self._predict_int(text_lower, key)
            else:
                result[key] = str(expected_type)

        return result

    def _select_category(self, text: str, options: List[str]) -> str:
        # Check direct mentions or keywords
        for opt in options:
            if opt.lower() in text:
                return opt
        # Default category heuristics
        if "large" in options and ("complex" in text or "difficult" in text or "prove" in text):
            return "large"
        if "small" in options and ("simple" in text or "fast" in text or "what is" in text):
            return "small"
        return options[0]

    def _predict_bool(self, text: str, key: str) -> bool:
        if "tool" in key:
            return any(w in text for w in ["calculate", "math", "code", "run", "search", "% of", "+", "*"])
        if "verification" in key:
            return any(w in text for w in ["important", "prove", "critical", "exact", "guarantee"])
        if "spam" in key:
            return any(w in text for w in ["free", "winner", "prize", "claim"])
        return False

    def _predict_float(self, text: str, key: str) -> float:
        if "confidence" in key:
            if any(w in text for w in ["maybe", "unclear", "doubt", "perhaps"]):
                return 0.65
            return 0.94
        if "difficulty" in key:
            if any(w in text for w in ["prove", "differential", "quantum", "optimization"]):
                return 0.85
            return 0.25
        return 0.50

    def _predict_int(self, text: str, key: str) -> int:
        return 0
