"""
AEON Phase 8: Hardware Memory Management & OOM Preflight Accounting.

Tracks unified memory vs discrete VRAM, performs rigorous preflight
resource checks to prevent Out-Of-Memory (OOM) failures, and generates
deterministic resource-aware fallback suggestions.
"""

from typing import Optional, Tuple
from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    MemoryPressure,
    MemoryAssessment,
    HardwareProfile,
)
from aeon.hardware.profiles import estimate_model_memory_mb


class HardwareMemoryManager:
    """
    Evaluates memory headroom, manages safety margins, and enforces
    OOM prevention across host RAM and accelerator VRAM.
    """

    def __init__(self, safety_margin_ratio: float = 0.15):
        self.safety_margin_ratio = max(0.01, min(0.50, safety_margin_ratio))

    def preflight_check(
        self,
        model_name: str,
        precision: PrecisionType,
        device: DeviceType,
        profile: HardwareProfile,
        seq_len: int = 512,
        memory_budget_mb: Optional[float] = None,
    ) -> MemoryAssessment:
        """
        Perform a non-invasive preflight memory assessment before launching computation.
        """
        # Determine available addressable memory for this device
        if device in [DeviceType.CUDA]:
            total_mb = profile.accelerator_memory_mb if profile.accelerator_memory_mb > 0 else 16384.0
            avail_mb = total_mb * (0.85 if profile.memory_pressure != MemoryPressure.HIGH else 0.40)
        elif profile.unified_memory or device == DeviceType.MPS:
            total_mb = profile.ram_total_mb
            avail_mb = profile.ram_available_mb
        else:
            total_mb = profile.ram_total_mb
            avail_mb = profile.ram_available_mb

        # User memory budget constraint clamp
        if memory_budget_mb is not None and memory_budget_mb > 0:
            avail_mb = min(avail_mb, memory_budget_mb)

        # Estimate required model memory
        model_mem_mb = estimate_model_memory_mb(model_name, precision, seq_len=seq_len)
        workspace_mb = round(model_mem_mb * 0.08, 2)
        kv_cache_mb = round(model_mem_mb * 0.05, 2)
        total_req_mb = model_mem_mb + workspace_mb + kv_cache_mb

        safety_margin_mb = round(total_req_mb * self.safety_margin_ratio, 2)
        effective_needed_mb = total_req_mb + safety_margin_mb
        headroom_mb = round(avail_mb - effective_needed_mb, 2)

        # OOM Risk calculation
        if avail_mb <= 0:
            oom_risk = 1.0
            fits = False
        else:
            ratio = total_req_mb / float(avail_mb)
            oom_risk = max(0.0, min(1.0, ratio))
            fits = (headroom_mb >= 0) and (ratio <= (1.0 - self.safety_margin_ratio))

        # Memory pressure classification
        avail_ratio = avail_mb / max(1.0, total_mb)
        if avail_ratio < 0.08 or headroom_mb < 0:
            pressure = MemoryPressure.CRITICAL
        elif avail_ratio < 0.20 or headroom_mb < 256.0:
            pressure = MemoryPressure.HIGH
        elif avail_ratio < 0.35:
            pressure = MemoryPressure.MODERATE
        else:
            pressure = MemoryPressure.NORMAL

        # Recommended fallback strategy
        fallback = None
        if not fits:
            if precision in [PrecisionType.FP32, PrecisionType.FP16, PrecisionType.BF16]:
                fallback = "downgrade_precision_to_int8"
            elif "70b" in model_name:
                fallback = "fallback_model_to_aeon-medium-7b"
            elif "7b" in model_name:
                fallback = "fallback_model_to_aeon-small-1.5b"
            else:
                fallback = "reduce_depth_and_mod_capacity"

        return MemoryAssessment(
            total_memory_mb=round(total_mb, 1),
            available_memory_mb=round(avail_mb, 1),
            reserved_memory_mb=round(total_mb - avail_mb, 1),
            estimated_model_mb=round(model_mem_mb, 1),
            estimated_kv_cache_mb=round(kv_cache_mb, 1),
            estimated_workspace_mb=round(workspace_mb, 1),
            safety_margin_mb=round(safety_margin_mb, 1),
            remaining_headroom_mb=round(headroom_mb, 1),
            memory_pressure=pressure,
            fits_in_memory=fits,
            oom_risk=round(oom_risk, 3),
            recommended_fallback=fallback,
        )
