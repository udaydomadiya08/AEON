"""
Abstract Inference Backend Interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from aeon.types import InferenceRequest, ExecutionPlan, Telemetry


class BaseInferenceBackend(ABC):
    """Abstract base class for all AEON model inference backends."""

    @abstractmethod
    def generate(
        self,
        request: InferenceRequest,
        plan: ExecutionPlan,
    ) -> Dict[str, Any]:
        """
        Execute generation for a request according to the execution plan.

        Returns a dict containing:
        - "answer": str
        - "input_tokens": int
        - "output_tokens": int
        - "reasoning_tokens": int
        - "latency_ms": float
        - "confidence": float
        - "compute_estimate": float (FLOPs)
        - "memory_mb": float
        - "estimated_energy": float (Joules)
        - "estimated_cost": float (USD)
        - "speculative": bool
        - "verification": bool
        - "escalated": bool
        - "correct": Optional[bool]
        """
        pass

    def supports_speculation(self) -> bool:
        """Return whether backend supports speculative decoding."""
        return True

    def supports_draft_model(self) -> bool:
        """Return whether backend supports draft model speculation."""
        return True

    def supports_ngram(self) -> bool:
        """Return whether backend supports prompt/context n-gram speculation."""
        return True

    def supports_multi_token_prediction(self) -> bool:
        """Return whether backend supports multi-token prediction heads."""
        return False

    def generate_draft(self, draft_model: str, prefix: str, draft_length: int) -> list:
        """Draft candidate tokens using specified draft model or heuristic."""
        return []

    def verify_speculation(self, target_model: str, prefix: str, draft_tokens: list) -> Dict[str, Any]:
        """Verify draft tokens against target model logits/distribution."""
        return {"accepted_tokens": draft_tokens, "rejected_tokens": []}

    def stream(self, request: InferenceRequest, plan: ExecutionPlan):
        """Stream token chunks for the request."""
        res = self.generate(request, plan)
        words = res["answer"].split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word

    def health(self) -> Dict[str, Any]:
        """Return backend health and availability metadata."""
        return {"status": "healthy", "available": True}

    def metadata(self) -> Dict[str, Any]:
        """Return backend architectural and engine metadata."""
        return {"type": self.__class__.__name__, "streaming_supported": True}

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        """Estimate monetary cost in USD for a given token volume."""
        return (prompt_tokens + completion_tokens) * 1e-6
