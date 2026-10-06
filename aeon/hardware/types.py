"""
AEON Phase 8: Hardware, Device, Precision & Quantization Types.

Defines first-class types for hardware detection, capability tracking,
memory accounting, precision levels, quantization methods, thermal states,
energy modeling, and economic hardware candidate evaluation.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import time
import uuid


class DeviceType(str, Enum):
    """Execution device / accelerator class."""
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    ROCM = "rocm"
    UNKNOWN = "unknown"


class PrecisionType(str, Enum):
    """Numerical representations for model weights and activations."""
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    FP8 = "FP8"


class QuantizationMethod(str, Enum):
    """Quantization methods supported across real and simulated backends."""
    NONE = "none"
    INT8_WEIGHT_ONLY = "int8_weight_only"
    INT8_DYNAMIC = "int8_dynamic"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    AWQ = "awq"
    GPTQ = "gptq"
    SIMULATED_INT8 = "simulated_int8"
    SIMULATED_FP8 = "simulated_fp8"


class CapabilityStatus(str, Enum):
    """Provenance and certainty status for a hardware or precision feature."""
    DETECTED = "detected"
    AVAILABLE = "available"
    SUPPORTED = "supported"
    OPTIMIZED = "optimized"
    MEASURED = "measured"
    ESTIMATED = "estimated"
    SIMULATED = "simulated"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ThermalPressure(str, Enum):
    """Thermal throttling status of host machine."""
    NOMINAL = "nominal"
    MODERATE = "moderate"
    HEAVY = "heavy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MemoryPressure(str, Enum):
    """Host or accelerator RAM pressure categorization."""
    NORMAL = "normal"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionMode(str, Enum):
    """Execution provenance mode."""
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    SIMULATED = "SIMULATED"
    HYBRID = "HYBRID"


class PrecisionProfile(BaseModel):
    """Characteristics and scaling factors of a specific numerical precision."""
    precision: PrecisionType
    bits: int
    numeric_family: str
    memory_factor: float
    compute_factor: float
    quality_delta: float
    status: CapabilityStatus = CapabilityStatus.ESTIMATED
    provenance: str = "analytical_scaling"


class HardwareProfile(BaseModel):
    """
    Comprehensive representation of discovered or simulated host hardware.
    Never exposes sensitive identifiers (MAC address, serials, user paths).
    """
    profile_id: str = Field(default_factory=lambda: f"hw-{uuid.uuid4().hex[:8]}")
    timestamp: float = Field(default_factory=time.time)
    os_name: str
    machine_arch: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_mb: float
    ram_available_mb: float
    accelerator_type: DeviceType
    accelerator_name: str
    accelerator_count: int = 1
    accelerator_memory_mb: float = 0.0
    unified_memory: bool = False
    pytorch_available: bool = False
    pytorch_version: str = "none"
    mps_available: bool = False
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_compute_capability: Optional[str] = None
    supported_precisions: Dict[str, str] = Field(default_factory=dict)
    quantization_support: Dict[str, str] = Field(default_factory=dict)
    memory_pressure: MemoryPressure = MemoryPressure.NORMAL
    thermal_pressure: ThermalPressure = ThermalPressure.NOMINAL
    thermal_temperature_c: Optional[float] = None
    power_measurement_supported: bool = False
    is_simulated: bool = False
    provenance: Dict[str, Any] = Field(default_factory=dict)


class QuantizationConfig(BaseModel):
    """Parameters governing model quantization."""
    enabled: bool = False
    method: QuantizationMethod = QuantizationMethod.NONE
    source_precision: PrecisionType = PrecisionType.FP16
    target_precision: PrecisionType = PrecisionType.INT8
    calibrated: bool = False
    calibration_dataset: Optional[str] = None
    per_channel: bool = True
    zero_point: bool = True


class QuantizationResult(BaseModel):
    """Outcome of real or simulated quantization."""
    config: QuantizationConfig
    memory_reduction_factor: float
    quality_impact: float
    latency_factor: float
    status: CapabilityStatus
    is_simulated: bool = True
    provenance: str = "SIMULATED_QUANTIZATION"


class MemoryAssessment(BaseModel):
    """
    Preflight resource assessment to ensure execution fits within memory
    and prevents Out-Of-Memory (OOM) failures.
    """
    total_memory_mb: float
    available_memory_mb: float
    reserved_memory_mb: float
    estimated_model_mb: float
    estimated_kv_cache_mb: float
    estimated_workspace_mb: float
    safety_margin_mb: float
    remaining_headroom_mb: float
    memory_pressure: MemoryPressure
    fits_in_memory: bool
    oom_risk: float = Field(ge=0.0, le=1.0)
    recommended_fallback: Optional[str] = None

    @property
    def fits(self) -> bool:
        return self.fits_in_memory


class ThermalState(BaseModel):
    """Host or accelerator thermal reading."""
    pressure: ThermalPressure = ThermalPressure.NOMINAL
    temperature_c: Optional[float] = None
    throttling_active: bool = False
    power_mode: Optional[str] = None
    status: CapabilityStatus = CapabilityStatus.ESTIMATED


class EnergyEstimate(BaseModel):
    """Power and energy estimates for computational actions."""
    energy_joules: float
    power_watts: float
    latency_s: float
    measurement_mode: ExecutionMode = ExecutionMode.ESTIMATED
    confidence: float = 0.85

    @property
    def joules(self) -> float:
        return self.energy_joules

    @property
    def watts(self) -> float:
        return self.power_watts

    @property
    def measurement_status(self) -> ExecutionMode:
        return self.measurement_mode


class HardwareCandidate(BaseModel):
    """
    Candidate execution tuple (Model x Device x Precision x Quantization x Depth)
    evaluated by the HardwareController.
    """
    candidate_id: str
    model_name: str
    device: DeviceType
    precision: PrecisionType
    quantization: QuantizationMethod
    depth: Optional[int] = None
    mod_capacity: Optional[float] = None
    estimated_quality: float
    estimated_latency_ms: float
    estimated_cost_usd: float
    estimated_memory_mb: float
    estimated_energy_joules: float
    estimated_flops: float
    memory_pressure: MemoryPressure
    thermal_risk: float
    precision_risk: float
    utility: float
    is_viable: bool
    fits_memory: bool = True
    rejection_reason: Optional[str] = None


class HardwareDecision(BaseModel):
    """
    Final selected hardware and precision configuration with complete provenance.
    """
    decision_id: str = Field(default_factory=lambda: f"hwd-{uuid.uuid4().hex[:8]}")
    timestamp: float = Field(default_factory=time.time)
    device: DeviceType
    precision: PrecisionType
    quantization: QuantizationMethod
    model_name: str
    memory_mb: float
    latency_ms: float
    cost_usd: float
    energy_joules: float
    quality_estimate: float
    utility: float
    execution_mode: ExecutionMode
    candidates_evaluated: int = 1
    candidates: List[HardwareCandidate] = Field(default_factory=list)
    selected_candidate: Optional[HardwareCandidate] = None
    fallbacks_triggered: List[str] = Field(default_factory=list)
    decision_reason: str = "economic_utility_optimal"
    hardware_profile_id: str = "unknown"
    hardware_profile: Optional[HardwareProfile] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)

    @property
    def selected(self) -> Optional[HardwareCandidate]:
        return self.selected_candidate


class HardwareConfig(BaseModel):
    """Configuration options for AEON Hardware Subsystem."""
    enabled: bool = True
    auto_detect: bool = True
    preferred_device: str = "auto"
    preferred_precision: str = "auto"
    preferred_quantization: str = "auto"
    memory_safety_margin: float = 0.15
    thermal_awareness: bool = True
    energy_awareness: bool = True
    allow_simulation: bool = True
    target_quality: float = 0.85
    cost_lambda: float = 10.0
    latency_lambda: float = 0.005
    memory_lambda: float = 0.0001
    energy_lambda: float = 0.05
    thermal_lambda: float = 0.5
    precision_risk_lambda: float = 0.3
