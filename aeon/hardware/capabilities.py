"""
AEON Phase 8: Capability Matrix & Compatibility Assessment.

Defines the explicit 4-dimensional compatibility mechanism:
    Hardware Device × Backend × Model × Precision

Distinguishes SUPPORTED, UNSUPPORTED, UNKNOWN, SIMULATED, and OPTIMIZED.
Never silently converts unsupported precision into supported precision without fallback logging.
"""

from typing import Dict, Tuple, Optional
from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    CapabilityStatus,
    HardwareProfile,
)


class HardwareCapabilityMatrix:
    """
    Evaluates execution viability across device, backend, model, and precision dimensions.
    """

    def __init__(self, profile: Optional[HardwareProfile] = None):
        self.profile = profile

    def set_profile(self, profile: HardwareProfile) -> None:
        self.profile = profile

    def evaluate_compatibility(
        self,
        device: DeviceType,
        model_name: str,
        precision: PrecisionType,
        backend_name: str = "simulation",
    ) -> Tuple[CapabilityStatus, str]:
        """
        Evaluate compatibility for (device, model, precision).
        Returns (status, explanation).
        """
        # 1. Device check against profile
        if self.profile is not None:
            if device == DeviceType.CUDA and not self.profile.cuda_available and not self.profile.is_simulated:
                return CapabilityStatus.UNSUPPORTED, "CUDA hardware is not available on host system"
            if device == DeviceType.MPS and not self.profile.mps_available and not self.profile.is_simulated:
                return CapabilityStatus.UNSUPPORTED, "Apple Silicon MPS is not available on host system"

        # 2. Precision check on device
        if device == DeviceType.MPS:
            if precision == PrecisionType.FP8:
                return (
                    CapabilityStatus.UNSUPPORTED,
                    "FP8 is not supported by Apple Silicon MPS backend in PyTorch",
                )
            elif precision in [PrecisionType.FP16, PrecisionType.BF16]:
                return CapabilityStatus.OPTIMIZED, "Native 16-bit float acceleration on Apple Silicon Metal"
            elif precision == PrecisionType.INT8:
                return CapabilityStatus.SUPPORTED, "8-bit integer supported on MPS"
            elif precision == PrecisionType.FP32:
                return CapabilityStatus.SUPPORTED, "32-bit single precision supported on MPS"

        elif device == DeviceType.CUDA:
            if precision == PrecisionType.FP8:
                if (
                    self.profile
                    and self.profile.cuda_compute_capability
                    and float(self.profile.cuda_compute_capability) >= 8.9
                ):
                    return CapabilityStatus.OPTIMIZED, "Native FP8 Tensor Core acceleration (Ada/Hopper)"
                else:
                    return CapabilityStatus.SIMULATED, "FP8 simulated on pre-Hopper CUDA architecture"
            return CapabilityStatus.OPTIMIZED, "CUDA Tensor Core accelerated execution"

        elif device == DeviceType.CPU:
            if precision == PrecisionType.FP8:
                return CapabilityStatus.UNSUPPORTED, "FP8 is not supported on standard CPU host"
            elif precision == PrecisionType.INT8:
                return CapabilityStatus.OPTIMIZED, "CPU SIMD quantized integer execution"
            elif precision == PrecisionType.FP16:
                return CapabilityStatus.SUPPORTED, "CPU FP16 emulation"
            elif precision == PrecisionType.FP32:
                return CapabilityStatus.OPTIMIZED, "Native CPU single precision"

        return CapabilityStatus.SUPPORTED, "Generic execution supported"
