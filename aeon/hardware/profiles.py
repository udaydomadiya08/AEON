"""
AEON Phase 8: Standard Hardware & Model Memory Profiles.

Provides standard hardware profiles for Apple Silicon, NVIDIA CUDA (A100, H100),
and CPU server nodes, along with analytical model memory footprint estimators.
"""

from typing import Dict, Optional, Any
from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    CapabilityStatus,
    ThermalPressure,
    MemoryPressure,
    HardwareProfile,
)


APPLE_SILICON_M_PROFILE = HardwareProfile(
    profile_id="hw-apple-silicon-m-series",
    os_name="Darwin",
    machine_arch="arm64",
    cpu_model="Apple Silicon (ARM64)",
    cpu_cores_physical=8,
    cpu_cores_logical=8,
    ram_total_mb=16384.0,
    ram_available_mb=8192.0,
    accelerator_type=DeviceType.MPS,
    accelerator_name="Apple Neural & Metal GPU (MPS)",
    accelerator_count=1,
    accelerator_memory_mb=16384.0,
    unified_memory=True,
    pytorch_available=True,
    pytorch_version="2.8.0",
    mps_available=True,
    cuda_available=False,
    supported_precisions={
        PrecisionType.FP32.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.FP16.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.BF16.value: CapabilityStatus.SUPPORTED.value,
        PrecisionType.INT8.value: CapabilityStatus.SUPPORTED.value,
        PrecisionType.FP8.value: CapabilityStatus.UNSUPPORTED.value,
    },
    quantization_support={
        "int8_weight_only": CapabilityStatus.SUPPORTED.value,
        "simulated_int8": CapabilityStatus.SIMULATED.value,
        "simulated_fp8": CapabilityStatus.SIMULATED.value,
    },
    memory_pressure=MemoryPressure.NORMAL,
    thermal_pressure=ThermalPressure.NOMINAL,
    is_simulated=False,
    provenance={"environment": "apple_silicon_metal"},
)

NVIDIA_A100_PROFILE = HardwareProfile(
    profile_id="hw-nvidia-a100-80gb",
    os_name="Linux",
    machine_arch="x86_64",
    cpu_model="AMD EPYC 7763 64-Core Processor",
    cpu_cores_physical=32,
    cpu_cores_logical=64,
    ram_total_mb=262144.0,
    ram_available_mb=196608.0,
    accelerator_type=DeviceType.CUDA,
    accelerator_name="NVIDIA A100-SXM4-80GB",
    accelerator_count=1,
    accelerator_memory_mb=81920.0,
    unified_memory=False,
    pytorch_available=True,
    pytorch_version="2.4.0",
    mps_available=False,
    cuda_available=True,
    cuda_device_count=1,
    cuda_compute_capability="8.0",
    supported_precisions={
        PrecisionType.FP32.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.FP16.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.BF16.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.INT8.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.FP8.value: CapabilityStatus.SIMULATED.value,
    },
    quantization_support={
        "int8_weight_only": CapabilityStatus.OPTIMIZED.value,
        "int8_dynamic": CapabilityStatus.OPTIMIZED.value,
        "simulated_fp8": CapabilityStatus.SIMULATED.value,
    },
    memory_pressure=MemoryPressure.NORMAL,
    thermal_pressure=ThermalPressure.NOMINAL,
    is_simulated=True,
    provenance={"environment": "datacenter_gpu_cluster"},
)

NVIDIA_H100_PROFILE = HardwareProfile(
    profile_id="hw-nvidia-h100-80gb",
    os_name="Linux",
    machine_arch="x86_64",
    cpu_model="Intel Xeon Platinum 8480+",
    cpu_cores_physical=56,
    cpu_cores_logical=112,
    ram_total_mb=524288.0,
    ram_available_mb=400000.0,
    accelerator_type=DeviceType.CUDA,
    accelerator_name="NVIDIA H100 80GB HBM3",
    accelerator_count=1,
    accelerator_memory_mb=81920.0,
    unified_memory=False,
    pytorch_available=True,
    pytorch_version="2.5.0",
    mps_available=False,
    cuda_available=True,
    cuda_device_count=1,
    cuda_compute_capability="9.0",
    supported_precisions={
        PrecisionType.FP32.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.FP16.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.BF16.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.INT8.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.FP8.value: CapabilityStatus.OPTIMIZED.value,  # Native Hopper Transformer Engine
    },
    quantization_support={
        "fp8_e4m3": CapabilityStatus.OPTIMIZED.value,
        "fp8_e5m2": CapabilityStatus.OPTIMIZED.value,
        "int8_weight_only": CapabilityStatus.OPTIMIZED.value,
    },
    memory_pressure=MemoryPressure.NORMAL,
    thermal_pressure=ThermalPressure.NOMINAL,
    is_simulated=True,
    provenance={"environment": "hopper_transformer_engine"},
)

CPU_SERVER_PROFILE = HardwareProfile(
    profile_id="hw-cpu-server-x86",
    os_name="Linux",
    machine_arch="x86_64",
    cpu_model="Intel Xeon Gold 6338",
    cpu_cores_physical=16,
    cpu_cores_logical=32,
    ram_total_mb=65536.0,
    ram_available_mb=49152.0,
    accelerator_type=DeviceType.CPU,
    accelerator_name="Host CPU Node (No Accelerator)",
    accelerator_count=0,
    accelerator_memory_mb=0.0,
    unified_memory=False,
    pytorch_available=True,
    pytorch_version="2.4.0",
    mps_available=False,
    cuda_available=False,
    supported_precisions={
        PrecisionType.FP32.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.FP16.value: CapabilityStatus.SUPPORTED.value,
        PrecisionType.BF16.value: CapabilityStatus.SUPPORTED.value,
        PrecisionType.INT8.value: CapabilityStatus.OPTIMIZED.value,
        PrecisionType.FP8.value: CapabilityStatus.UNSUPPORTED.value,
    },
    quantization_support={
        "int8_weight_only": CapabilityStatus.SUPPORTED.value,
        "simulated_int8": CapabilityStatus.SIMULATED.value,
    },
    memory_pressure=MemoryPressure.NORMAL,
    thermal_pressure=ThermalPressure.NOMINAL,
    is_simulated=True,
    provenance={"environment": "cpu_only_server"},
)

# Standard parameter counts in billions
MODEL_PARAM_COUNTS: Dict[str, float] = {
    "aeon-decision-jev": 0.08,
    "aeon-draft-0.5b": 0.50,
    "aeon-small-1.5b": 1.50,
    "aeon-medium-7b": 7.00,
    "aeon-large-70b": 70.00,
}


def estimate_model_memory_mb(
    model_name: str,
    precision: PrecisionType,
    seq_len: int = 512,
    quantization: Optional[Any] = None,
) -> float:
    """
    Analytically compute the memory footprint of model weights and KV cache.
    Weight memory: params_b * (bits / 8) * 1024 MB
    KV Cache: 2 * layers * hidden_dim * seq_len * (bits / 8)
    """
    if not isinstance(seq_len, int):
        quantization = seq_len
        seq_len = 512

    params = MODEL_PARAM_COUNTS.get(model_name, 7.0)
    bits = 32
    if precision in [PrecisionType.FP16, PrecisionType.BF16]:
        bits = 16
    elif precision in [PrecisionType.INT8, PrecisionType.FP8]:
        bits = 8

    # Quantization impacts effective bit precision of weights
    if quantization is not None:
        q_val = quantization.value if hasattr(quantization, "value") else str(quantization)
        if "int4" in q_val:
            bits = min(bits, 4)
        elif "int8" in q_val or "fp8" in q_val:
            bits = min(bits, 8)

    # Weights
    weights_mb = params * (bits / 8.0) * 1024.0

    # Approximate KV cache & workspace
    layers = 32
    if "70b" in model_name:
        layers = 80
    elif "1.5b" in model_name:
        layers = 28
    elif "draft" in model_name or "decision" in model_name:
        layers = 16

    hidden_dim = 4096 if "70b" not in model_name else 8192
    kv_cache_mb = (2 * layers * hidden_dim * seq_len * (bits / 8.0)) / (1024.0 * 1024.0)
    workspace_mb = min(512.0, weights_mb * 0.05)

    return round(weights_mb + kv_cache_mb + workspace_mb, 2)
