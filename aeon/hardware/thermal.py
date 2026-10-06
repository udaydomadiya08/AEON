"""
AEON Phase 8: Thermal Monitoring & Throttling Awareness.

Monitors host machine and accelerator thermal pressure levels
and translates thermal risk into economic throttling penalties.
"""

from typing import Optional
from aeon.hardware.types import (
    ThermalPressure,
    CapabilityStatus,
    ThermalState,
)


class ThermalMonitor:
    """
    Monitors thermal pressure and recommends adaptive compute mitigation
    to prevent aggressive thermal throttling.
    """

    def __init__(self):
        pass

    def evaluate_thermal_risk(self, state: ThermalState) -> float:
        """
        Quantify thermal risk as a normalized penalty factor in [0.0, 1.0].
        """
        if state.pressure == ThermalPressure.CRITICAL:
            return 1.0
        elif state.pressure == ThermalPressure.HEAVY:
            return 0.70
        elif state.pressure == ThermalPressure.MODERATE:
            return 0.30
        elif state.pressure == ThermalPressure.NOMINAL:
            return 0.0
        return 0.10

    def recommend_thermal_mitigation(self, state: ThermalState) -> Optional[str]:
        """
        Generate operational recommendations when host is under thermal strain.
        """
        if state.pressure in [ThermalPressure.CRITICAL, ThermalPressure.HEAVY]:
            return "throttle_depth_and_enable_int8_quantization"
        elif state.pressure == ThermalPressure.MODERATE:
            return "favor_fp16_over_fp32"
        return None
