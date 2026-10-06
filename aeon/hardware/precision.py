"""
AEON Phase 8: Precision System & Numerical Representations.

Provides first-class precision abstractions (FP32, FP16, BF16, INT8, FP8),
hardware compatibility assessment, quality degradation modeling,
and deterministic, provenance-backed safe fallback chains.
"""

from typing import Dict, List, Optional, Tuple, Union
from aeon.hardware.types import (
    PrecisionType,
    DeviceType,
    CapabilityStatus,
    PrecisionProfile,
)


STANDARD_PRECISION_PROFILES: Dict[PrecisionType, PrecisionProfile] = {
    PrecisionType.FP32: PrecisionProfile(
        precision=PrecisionType.FP32,
        bits=32,
        numeric_family="IEEE-754 Single Precision",
        memory_factor=1.0,
        compute_factor=1.0,
        quality_delta=0.0,
        status=CapabilityStatus.SUPPORTED,
        provenance="baseline_standard",
    ),
    PrecisionType.FP16: PrecisionProfile(
        precision=PrecisionType.FP16,
        bits=16,
        numeric_family="IEEE-754 Half Precision",
        memory_factor=0.50,
        compute_factor=0.55,
        quality_delta=-0.001,
        status=CapabilityStatus.SUPPORTED,
        provenance="hardware_accelerated_half",
    ),
    PrecisionType.BF16: PrecisionProfile(
        precision=PrecisionType.BF16,
        bits=16,
        numeric_family="Brain Floating Point 16",
        memory_factor=0.50,
        compute_factor=0.55,
        quality_delta=-0.001,
        status=CapabilityStatus.SUPPORTED,
        provenance="bfloat16_dynamic_range",
    ),
    PrecisionType.INT8: PrecisionProfile(
        precision=PrecisionType.INT8,
        bits=8,
        numeric_family="8-bit Signed Integer (Quantized)",
        memory_factor=0.25,
        compute_factor=0.35,
        quality_delta=-0.012,
        status=CapabilityStatus.ESTIMATED,
        provenance="quantized_integer_arithmetic",
    ),
    PrecisionType.FP8: PrecisionProfile(
        precision=PrecisionType.FP8,
        bits=8,
        numeric_family="8-bit Floating Point (E4M3/E5M2)",
        memory_factor=0.25,
        compute_factor=0.30,
        quality_delta=-0.006,
        status=CapabilityStatus.SIMULATED,
        provenance="simulated_fp8_transformer_engine",
    ),
}

# Deterministic fallback hierarchies
PRECISION_FALLBACK_CHAINS: Dict[PrecisionType, List[PrecisionType]] = {
    PrecisionType.FP8: [PrecisionType.BF16, PrecisionType.FP16, PrecisionType.FP32],
    PrecisionType.INT8: [PrecisionType.FP16, PrecisionType.BF16, PrecisionType.FP32],
    PrecisionType.BF16: [PrecisionType.FP16, PrecisionType.FP32],
    PrecisionType.FP16: [PrecisionType.FP32],
    PrecisionType.FP32: [],
}


class PrecisionManager:
    """
    Manages numerical precision compatibility, task-sensitive quality
    degradation, and safe deterministic fallback resolution.
    """

    def __init__(self, profiles: Optional[Dict[PrecisionType, PrecisionProfile]] = None):
        self.profiles = profiles or dict(STANDARD_PRECISION_PROFILES)

    def get_profile(self, precision: Union[PrecisionType, str]) -> PrecisionProfile:
        p_enum = PrecisionType(precision) if isinstance(precision, str) else precision
        return self.profiles.get(p_enum, self.profiles[PrecisionType.FP32])

    def estimate_quality_impact(
        self,
        precision: PrecisionType,
        base_quality: float,
        difficulty: float = 0.5,
        model_name: str = "aeon-medium-7b",
    ) -> float:
        """
        Estimate the expected quality of an execution given precision degradation,
        task difficulty, and model parameter resilience.
        """
        prof = self.get_profile(precision)
        delta = prof.quality_delta

        # Harder tasks are more sensitive to quantization noise
        difficulty_penalty = difficulty * abs(delta) * 0.80

        # Larger models are empirically more resilient to quantization noise
        resilience = 1.0
        if "70b" in model_name.lower():
            resilience = 0.60
        elif "1.5b" in model_name.lower():
            resilience = 1.40

        net_delta = delta * resilience - difficulty_penalty
        return max(0.10, min(1.0, base_quality + net_delta))

    def resolve_fallback(
        self,
        requested_precision: PrecisionType,
        device: DeviceType,
        supported_precisions: Dict[str, str],
    ) -> Tuple[PrecisionType, List[str], str]:
        """
        Deterministically resolve the best supported precision given a requested
        precision and the target device's supported precision set.
        Returns (selected_precision, fallback_history, reason).
        """
        def _get_status(val: str) -> str:
            return (
                supported_precisions.get(val)
                or supported_precisions.get(val.upper())
                or supported_precisions.get(val.lower())
                or CapabilityStatus.UNSUPPORTED.value
            )

        req_str = requested_precision.value
        status = _get_status(req_str)

        # If requested precision is supported or simulated
        if status in [
            CapabilityStatus.SUPPORTED.value,
            CapabilityStatus.AVAILABLE.value,
            CapabilityStatus.OPTIMIZED.value,
            CapabilityStatus.SIMULATED.value,
        ]:
            return requested_precision, [], "requested_precision_available"

        # Walk the fallback chain
        chain = PRECISION_FALLBACK_CHAINS.get(requested_precision, [PrecisionType.FP32])
        fallbacks = [f"requested_{req_str}_unsupported_on_{device.value}"]

        for cand in chain:
            cand_status = _get_status(cand.value)
            fallbacks.append(f"try_{cand.value}")
            if cand_status in [
                CapabilityStatus.SUPPORTED.value,
                CapabilityStatus.AVAILABLE.value,
                CapabilityStatus.OPTIMIZED.value,
                CapabilityStatus.SIMULATED.value,
            ]:
                return cand, fallbacks, f"fallback_to_{cand.value}_from_{req_str}"

        # Default safe fallback is always FP32
        fallbacks.append(f"ultimate_fallback_{PrecisionType.FP32.value}")
        return PrecisionType.FP32, fallbacks, "default_fp32_fallback"
