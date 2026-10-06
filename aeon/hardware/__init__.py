"""
AEON Phase 8: Hardware, Device, Precision & Quantization Subsystem.

Exports public types, profiles, detectors, managers, and controllers.
"""

from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    QuantizationMethod,
    CapabilityStatus,
    ThermalPressure,
    MemoryPressure,
    ExecutionMode,
    HardwareProfile,
    PrecisionProfile,
    QuantizationConfig,
    QuantizationResult,
    MemoryAssessment,
    ThermalState,
    EnergyEstimate,
    HardwareCandidate,
    HardwareDecision,
    HardwareConfig,
)
from aeon.hardware.detector import HardwareDetector
from aeon.hardware.profiles import (
    APPLE_SILICON_M_PROFILE,
    NVIDIA_A100_PROFILE,
    NVIDIA_H100_PROFILE,
    CPU_SERVER_PROFILE,
    estimate_model_memory_mb,
)
from aeon.hardware.precision import PrecisionManager
from aeon.hardware.quantization import QuantizationManager
from aeon.hardware.capabilities import HardwareCapabilityMatrix
from aeon.hardware.memory import HardwareMemoryManager
from aeon.hardware.cost_model import HardwareCostModel
from aeon.hardware.energy import EnergyEstimator
from aeon.hardware.thermal import ThermalMonitor
from aeon.hardware.backend import (
    BaseHardwareBackend,
    SimulationHardwareBackend,
)
from aeon.hardware.controller import (
    HardwareController,
    global_hardware_controller,
)
from aeon.hardware.telemetry import extract_hardware_telemetry

__all__ = [
    "DeviceType",
    "PrecisionType",
    "QuantizationMethod",
    "CapabilityStatus",
    "ThermalPressure",
    "MemoryPressure",
    "ExecutionMode",
    "HardwareProfile",
    "PrecisionProfile",
    "QuantizationConfig",
    "QuantizationResult",
    "MemoryAssessment",
    "ThermalState",
    "EnergyEstimate",
    "HardwareCandidate",
    "HardwareDecision",
    "HardwareConfig",
    "HardwareDetector",
    "APPLE_SILICON_M_PROFILE",
    "NVIDIA_A100_PROFILE",
    "NVIDIA_H100_PROFILE",
    "CPU_SERVER_PROFILE",
    "estimate_model_memory_mb",
    "PrecisionManager",
    "QuantizationManager",
    "HardwareCapabilityMatrix",
    "HardwareMemoryManager",
    "HardwareCostModel",
    "EnergyEstimator",
    "ThermalMonitor",
    "BaseHardwareBackend",
    "SimulationHardwareBackend",
    "HardwareController",
    "global_hardware_controller",
    "extract_hardware_telemetry",
]
