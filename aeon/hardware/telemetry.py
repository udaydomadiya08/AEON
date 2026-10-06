"""
AEON Phase 8: Hardware Telemetry & Observability.

Translates HardwareDecision objects into structured execution trace fields
and provides privacy-preserving provenance logs.
"""

from typing import Dict, Any
from aeon.hardware.types import HardwareDecision


def extract_hardware_telemetry(decision: HardwareDecision) -> Dict[str, Any]:
    """
    Extract structured dictionary of hardware decision metrics suitable
    for inclusion in AEON Telemetry.
    """
    return {
        "hardware_device": decision.device.value,
        "hardware_precision": decision.precision.value,
        "hardware_quantization": decision.quantization.value,
        "hardware_memory_mb": decision.memory_mb,
        "hardware_energy_joules": decision.energy_joules,
        "hardware_utility": decision.utility,
        "hardware_candidates_evaluated": decision.candidates_evaluated,
        "hardware_fallbacks_triggered": decision.fallbacks_triggered,
        "hardware_decision_reason": decision.decision_reason,
        "hardware_controller_overhead_ms": decision.provenance.get("controller_overhead_ms", 0.0),
        "hardware_execution_mode": decision.execution_mode.value,
    }
