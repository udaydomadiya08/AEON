"""
AEON Phase 7 — Early-Exit Evaluator.

Evaluates whether to terminate layer execution early or continue to subsequent layers.
Integrates calibration, confidence, verification requirements, and economic VOI.
"""

from typing import NamedTuple, Optional, Any
import math
from aeon.depth.types import DepthExitDecision, ModelDepthProfile, DepthConfig


class EarlyExitResult(NamedTuple):
    should_exit: bool
    decision: DepthExitDecision
    reason: str


class EarlyExitEvaluator:
    """
    Evaluates early-exit decisions at candidate layer checkpoints.
    Prevents unsafe premature exits and enforces verification gates.
    """

    def __init__(self, config: Optional[DepthConfig] = None):
        self.config = config or DepthConfig()

    def evaluate_exit(
        self,
        current_layer: int,
        profile: ModelDepthProfile,
        request: Any = None,
        confidence: Optional[float] = None,
        current_confidence: Optional[float] = None,
        current_quality: Optional[float] = None,
        estimated_quality: Optional[float] = None,
        marginal_gain: Optional[float] = None,
        marginal_quality_gain: Optional[float] = None,
        marginal_cost: Optional[float] = None,
        marginal_cost_usd: Optional[float] = None,
        voi: Optional[float] = None,
        require_verification: Optional[bool] = None,
        verified: bool = False,
        verification_passed: Optional[bool] = None,
        remaining_cost_usd: Optional[float] = None,
        remaining_latency_ms: Optional[float] = None,
        **kwargs,
    ) -> EarlyExitResult:
        """
        Evaluate whether model execution should exit at current_layer.
        Returns: EarlyExitResult(should_exit: bool, decision: DepthExitDecision, reason: str)
        """
        # Resolve confidence
        c = confidence if confidence is not None else current_confidence
        if c is None:
            c = 0.5
        # Check NaN / Inf
        if math.isnan(c) or math.isinf(c):
            return EarlyExitResult(False, DepthExitDecision.ERROR, "NaN/Inf confidence detected")

        # Resolve quality
        q = current_quality if current_quality is not None else estimated_quality
        if q is None:
            q = profile.get_quality_at_depth(current_layer)
        if math.isnan(q) or math.isinf(q):
            return EarlyExitResult(False, DepthExitDecision.ERROR, "NaN/Inf quality detected")

        # Resolve marginal gain and cost
        mg = marginal_gain if marginal_gain is not None else marginal_quality_gain
        if mg is None:
            mg = 0.005
        mc = marginal_cost if marginal_cost is not None else marginal_cost_usd
        if mc is None:
            mc = profile.layer_cost_usd

        # Resolve verification requirement
        req_verif = require_verification
        if req_verif is None and request is not None:
            req_verif = getattr(request, "require_verification", False)
        req_verif = bool(req_verif)

        is_verified = verified or bool(verification_passed)

        # Hard Invariant 7: Cannot exit before minimum safe depth
        if current_layer < profile.min_safe_depth:
            return EarlyExitResult(False, DepthExitDecision.CONTINUE, f"Below minimum safe depth ({profile.min_safe_depth})")

        # If already at full depth, exit
        if current_layer >= profile.total_layers:
            return EarlyExitResult(True, DepthExitDecision.EXIT, "Full depth reached")

        # Check binding budget constraints
        if remaining_cost_usd is not None and remaining_cost_usd <= 0.0:
            return EarlyExitResult(True, DepthExitDecision.BUDGET_EXHAUSTED, "Cost budget exhausted")

        if remaining_latency_ms is not None and remaining_latency_ms <= 0.0:
            return EarlyExitResult(True, DepthExitDecision.SLA_EXCEEDED, "Latency SLA exceeded")

        # Verification gate: tasks requiring verification cannot exit early unless verified
        if req_verif and not is_verified:
            verif_min_depth = int(round(profile.total_layers * 0.75))
            if current_layer < verif_min_depth:
                return EarlyExitResult(False, DepthExitDecision.CONTINUE, "Verification required: continue depth")

        target_q = kwargs.get("target_quality", None)
        if target_q is None:
            target_q = getattr(request, "target_quality", None) if request else None
        if target_q is None:
            target_q = self.config.target_quality

        diff = kwargs.get("difficulty", None)
        if diff is not None and diff <= 0.35:
            target_q = min(target_q, 0.70)

        # Quality target reached
        if q >= target_q:
            return EarlyExitResult(True, DepthExitDecision.QUALITY_SUFFICIENT, f"Quality {q:.2f} meets target {target_q:.2f}")

        # Target confidence reached
        if c >= self.config.confidence_threshold:
            return EarlyExitResult(True, DepthExitDecision.CONFIDENCE_SUFFICIENT, f"Confidence {c:.2f} exceeds target")

        # Economic VOI check: if marginal gain is negligible, exit
        if mg < 0.005:
            return EarlyExitResult(True, DepthExitDecision.LOW_MARGINAL_GAIN, f"Marginal gain {mg:.4f} below threshold")

        return EarlyExitResult(False, DepthExitDecision.CONTINUE, "Additional depth economically viable")


global_early_exit_evaluator = EarlyExitEvaluator()

