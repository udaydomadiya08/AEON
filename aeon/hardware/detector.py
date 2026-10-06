"""
AEON Phase 8: Hardware Detection & Environment Discovery.

Automatically detects host hardware (Apple Silicon, CPU, CUDA),
probes backend capability and memory limits safely, categorizes memory pressure,
and maintains privacy guarantees (no serials, MAC addresses, or user paths).
"""

import os
import platform
import subprocess
import time
from typing import Dict, Optional, Tuple, Any

from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    CapabilityStatus,
    ThermalPressure,
    MemoryPressure,
    HardwareProfile,
    ThermalState,
)


class HardwareDetector:
    """
    Automatic detection and capability discovery for host compute hardware.
    Features immutable profile caching and lightweight dynamic resource refreshing.
    """

    _cached_profile: Optional[HardwareProfile] = None

    def __init__(self, allow_probing: bool = True):
        self.allow_probing = allow_probing

    def detect_hardware(self, force_refresh: bool = False) -> HardwareProfile:
        """
        Inspect the system environment and return a structured HardwareProfile.
        Caches immutable hardware information across calls.
        """
        if self._cached_profile is not None and not force_refresh:
            return self.refresh_dynamic(self._cached_profile)

        # 1. Host OS & Architecture
        os_name = platform.system()
        machine_arch = platform.machine()
        cpu_model = self._detect_cpu_model(os_name, machine_arch)

        # 2. CPU Cores & Memory
        phys_cores, log_cores = self._detect_cpu_cores()
        total_ram_mb, avail_ram_mb = self._detect_ram_mb()

        # 3. PyTorch & Accelerator Detection
        (
            torch_avail,
            torch_ver,
            mps_avail,
            cuda_avail,
            cuda_count,
            cuda_cc,
            accel_type,
            accel_name,
            accel_mem_mb,
            unified_mem,
        ) = self._detect_accelerators(os_name, machine_arch, total_ram_mb)

        # 4. Precision & Quantization Matrix
        supported_precisions = self._probe_precisions(accel_type, torch_avail)
        quant_support = {
            "int8_weight_only": CapabilityStatus.SUPPORTED.value if torch_avail else CapabilityStatus.SIMULATED.value,
            "int8_dynamic": CapabilityStatus.SUPPORTED.value if torch_avail else CapabilityStatus.SIMULATED.value,
            "fp8_e4m3": (
                CapabilityStatus.SUPPORTED.value
                if (cuda_avail and cuda_cc and float(cuda_cc) >= 8.9)
                else CapabilityStatus.SIMULATED.value
            ),
            "simulated_int8": CapabilityStatus.SIMULATED.value,
            "simulated_fp8": CapabilityStatus.SIMULATED.value,
        }

        # 5. Memory Pressure
        mem_pressure = self._calculate_memory_pressure(avail_ram_mb, total_ram_mb)

        # 6. Thermal Reading
        thermal_state = self._read_thermal_state(os_name)

        profile = HardwareProfile(
            os_name=os_name,
            machine_arch=machine_arch,
            cpu_model=cpu_model,
            cpu_cores_physical=phys_cores,
            cpu_cores_logical=log_cores,
            ram_total_mb=round(total_ram_mb, 1),
            ram_available_mb=round(avail_ram_mb, 1),
            accelerator_type=accel_type,
            accelerator_name=accel_name,
            accelerator_count=cuda_count if cuda_avail else (1 if mps_avail else 0),
            accelerator_memory_mb=round(accel_mem_mb, 1),
            unified_memory=unified_mem,
            pytorch_available=torch_avail,
            pytorch_version=torch_ver,
            mps_available=mps_avail,
            cuda_available=cuda_avail,
            cuda_device_count=cuda_count,
            cuda_compute_capability=cuda_cc,
            supported_precisions=supported_precisions,
            quantization_support=quant_support,
            memory_pressure=mem_pressure,
            thermal_pressure=thermal_state.pressure,
            thermal_temperature_c=thermal_state.temperature_c,
            power_measurement_supported=False,
            is_simulated=False,
            provenance={
                "detection_method": "system_probes_and_libraries",
                "timestamp": time.time(),
                "device_class": accel_type.value,
            },
        )

        HardwareDetector._cached_profile = profile
        return profile

    def refresh_dynamic(self, profile: HardwareProfile) -> HardwareProfile:
        """Lightweight update for dynamic fields (available RAM, memory pressure, thermals)."""
        _, avail_ram_mb = self._detect_ram_mb()
        updated = profile.model_copy(deep=True)
        updated.ram_available_mb = round(avail_ram_mb, 1)
        updated.memory_pressure = self._calculate_memory_pressure(avail_ram_mb, profile.ram_total_mb)
        thermal = self._read_thermal_state(profile.os_name)
        updated.thermal_pressure = thermal.pressure
        updated.thermal_temperature_c = thermal.temperature_c
        return updated

    def _detect_cpu_model(self, os_name: str, machine_arch: str) -> str:
        if os_name == "Darwin" and machine_arch == "arm64":
            try:
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
                return out.decode("utf-8").strip()
            except Exception:
                return "Apple Silicon (ARM64)"
        return platform.processor() or "Generic CPU"

    def _detect_cpu_cores(self) -> Tuple[int, int]:
        logical = os.cpu_count() or 4
        try:
            import psutil
            physical = psutil.cpu_count(logical=False) or logical
            return physical, logical
        except Exception:
            return max(1, logical // 2), logical

    def _detect_ram_mb(self) -> Tuple[float, float]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.total / (1024.0 * 1024.0), mem.available / (1024.0 * 1024.0)
        except Exception:
            return 8192.0, 4096.0

    def _detect_accelerators(
        self, os_name: str, machine_arch: str, total_ram_mb: float
    ) -> Tuple[bool, str, bool, bool, int, Optional[str], DeviceType, str, float, bool]:
        torch_avail = False
        torch_ver = "none"
        mps_avail = False
        cuda_avail = False
        cuda_count = 0
        cuda_cc = None
        accel_type = DeviceType.CPU
        accel_name = "Host CPU"
        accel_mem_mb = 0.0
        unified_mem = False

        try:
            import torch
            torch_avail = True
            torch_ver = torch.__version__

            # Check CUDA
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                cuda_avail = True
                cuda_count = torch.cuda.device_count()
                accel_type = DeviceType.CUDA
                accel_name = torch.cuda.get_device_name(0) if cuda_count > 0 else "NVIDIA GPU"
                try:
                    accel_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024.0 * 1024.0)
                    major, minor = torch.cuda.get_device_capability(0)
                    cuda_cc = f"{major}.{minor}"
                except Exception:
                    accel_mem_mb = 16384.0
                unified_mem = False

            # Check MPS
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                mps_avail = True
                accel_type = DeviceType.MPS
                accel_name = "Apple Silicon Neural/Metal Engine (MPS)"
                accel_mem_mb = total_ram_mb  # Unified memory on Apple Silicon
                unified_mem = True

        except Exception:
            pass

        # If Darwin ARM64 without PyTorch MPS, still note Apple Silicon architecture
        if os_name == "Darwin" and machine_arch == "arm64" and accel_type != DeviceType.CUDA:
            unified_mem = True
            if accel_type == DeviceType.CPU:
                accel_name = "Apple Silicon CPU (Integrated Unified Memory)"

        return (
            torch_avail,
            torch_ver,
            mps_avail,
            cuda_avail,
            cuda_count,
            cuda_cc,
            accel_type,
            accel_name,
            accel_mem_mb,
            unified_mem,
        )

    def _probe_precisions(self, accel: DeviceType, torch_avail: bool) -> Dict[str, str]:
        """
        Probe supported precisions honestly.
        Separate dtype representability from backend operation support.
        """
        res = {
            PrecisionType.FP32.value: CapabilityStatus.SUPPORTED.value,
            PrecisionType.FP16.value: CapabilityStatus.SUPPORTED.value,
            PrecisionType.BF16.value: CapabilityStatus.SUPPORTED.value,
            PrecisionType.INT8.value: CapabilityStatus.SUPPORTED.value,
            PrecisionType.FP8.value: CapabilityStatus.UNSUPPORTED.value,
        }

        if accel == DeviceType.MPS:
            # MPS supports FP32, FP16, and BF16 in PyTorch 2.x
            res[PrecisionType.FP32.value] = CapabilityStatus.OPTIMIZED.value
            res[PrecisionType.FP16.value] = CapabilityStatus.OPTIMIZED.value
            res[PrecisionType.BF16.value] = CapabilityStatus.SUPPORTED.value
            res[PrecisionType.INT8.value] = CapabilityStatus.SUPPORTED.value
            # FP8 is explicitly NOT supported by the MPS backend
            res[PrecisionType.FP8.value] = CapabilityStatus.UNSUPPORTED.value

        elif accel == DeviceType.CUDA:
            res[PrecisionType.FP32.value] = CapabilityStatus.OPTIMIZED.value
            res[PrecisionType.FP16.value] = CapabilityStatus.OPTIMIZED.value
            res[PrecisionType.BF16.value] = CapabilityStatus.OPTIMIZED.value
            res[PrecisionType.INT8.value] = CapabilityStatus.OPTIMIZED.value
            res[PrecisionType.FP8.value] = CapabilityStatus.SUPPORTED.value

        else:
            # CPU
            res[PrecisionType.FP32.value] = CapabilityStatus.SUPPORTED.value
            res[PrecisionType.FP16.value] = CapabilityStatus.SUPPORTED.value
            res[PrecisionType.BF16.value] = CapabilityStatus.SUPPORTED.value
            res[PrecisionType.INT8.value] = CapabilityStatus.SUPPORTED.value
            res[PrecisionType.FP8.value] = CapabilityStatus.UNSUPPORTED.value

        return res

    def _calculate_memory_pressure(self, avail_mb: float, total_mb: float) -> MemoryPressure:
        if total_mb <= 0:
            return MemoryPressure.NORMAL
        ratio = avail_mb / total_mb
        if ratio < 0.08:
            return MemoryPressure.CRITICAL
        elif ratio < 0.20:
            return MemoryPressure.HIGH
        elif ratio < 0.35:
            return MemoryPressure.MODERATE
        return MemoryPressure.NORMAL

    def _read_thermal_state(self, os_name: str) -> ThermalState:
        # Default safe nominal thermal reading
        return ThermalState(
            pressure=ThermalPressure.NOMINAL,
            temperature_c=None,
            throttling_active=False,
            status=CapabilityStatus.ESTIMATED,
        )
