"""
AEON Phase 8: Hardware-Aware Cost, Latency & FLOP Economic Model.

Computes hardware- and precision-specific execution latency, FLOP counts,
energy, and monetary costs for candidate execution configurations.
"""

from typing import Dict, Tuple
from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    QuantizationMethod,
)
from aeon.hardware.profiles import MODEL_PARAM_COUNTS


# Device execution latency multiplier relative to reference GPU baseline
DEVICE_SPEED_FACTORS: Dict[DeviceType, float] = {
    DeviceType.CUDA: 0.85,
    DeviceType.MPS: 1.10,
    DeviceType.CPU: 3.20,
    DeviceType.ROCM: 1.00,
    DeviceType.UNKNOWN: 2.00,
}

# Precision latency scaling factors by device class
PRECISION_LATENCY_FACTORS: Dict[Tuple[DeviceType, PrecisionType], float] = {
    # CUDA
    (DeviceType.CUDA, PrecisionType.FP32): 1.00,
    (DeviceType.CUDA, PrecisionType.FP16): 0.52,
    (DeviceType.CUDA, PrecisionType.BF16): 0.52,
    (DeviceType.CUDA, PrecisionType.INT8): 0.35,
    (DeviceType.CUDA, PrecisionType.FP8): 0.30,
    # MPS
    (DeviceType.MPS, PrecisionType.FP32): 1.00,
    (DeviceType.MPS, PrecisionType.FP16): 0.55,
    (DeviceType.MPS, PrecisionType.BF16): 0.55,
    (DeviceType.MPS, PrecisionType.INT8): 0.48,
    (DeviceType.MPS, PrecisionType.FP8): 1.00,  # Unsupported, falls back
    # CPU
    (DeviceType.CPU, PrecisionType.FP32): 1.00,
    (DeviceType.CPU, PrecisionType.FP16): 0.85,
    (DeviceType.CPU, PrecisionType.BF16): 0.85,
    (DeviceType.CPU, PrecisionType.INT8): 0.60,
    (DeviceType.CPU, PrecisionType.FP8): 1.00,
}


class HardwareCostModel:
    """
    Computes hardware-grounded execution metrics including latency,
    FLOPs, and economic monetary cost.
    """

    def __init__(self, electricity_cost_kwh_usd: float = 0.15):
        # Electricity cost per Joule ($0.15 / 3,600,000 J)
        self.cost_per_joule_usd = electricity_cost_kwh_usd / 3600000.0

    def compute_metrics(
        self,
        model_name: str,
        device: DeviceType,
        precision: PrecisionType,
        quantization: QuantizationMethod,
        tokens_count: int = 100,
        layer_ratio: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute latency (ms), FLOPs, and monetary cost (USD) for a configuration.
        """
        params_b = MODEL_PARAM_COUNTS.get(model_name, 7.0)

        # Baseline latency per token in ms
        base_ms_per_token = 0.05 + (params_b * 0.035)
        dev_factor = DEVICE_SPEED_FACTORS.get(device, 1.50)
        prec_factor = PRECISION_LATENCY_FACTORS.get((device, precision), 0.60)

        if quantization in [QuantizationMethod.INT8_WEIGHT_ONLY, QuantizationMethod.SIMULATED_INT8]:
            prec_factor *= 0.90

        # Latency in ms
        latency_ms = max(
            0.5,
            tokens_count * base_ms_per_token * dev_factor * prec_factor * layer_ratio,
        )

        # Theoretical FLOPs: 2 * tokens * params * layer_ratio
        flops = 2.0 * tokens_count * (params_b * 1e9) * layer_ratio

        # Precision operations factor for effective compute
        if precision in [PrecisionType.INT8, PrecisionType.FP8]:
            effective_flops = flops * 0.50
        elif precision in [PrecisionType.FP16, PrecisionType.BF16]:
            effective_flops = flops * 0.75
        else:
            effective_flops = flops

        # Monetary Cost Model:
        # Base token rate amortized + local electricity energy cost
        cloud_token_rate = (params_b * 0.035) / 1000000.0  # Cost per token
        if device in [DeviceType.MPS, DeviceType.CPU]:
            # Local on-premise execution has zero API cloud markups
            # Cost reflects local electricity consumption + hardware depreciation
            local_depreciation = tokens_count * (params_b * 0.002) / 1000000.0
            cost_usd = local_depreciation + (latency_ms / 1000.0 * 35.0 * self.cost_per_joule_usd)
        else:
            cost_usd = (tokens_count * cloud_token_rate * prec_factor * layer_ratio)

        return {
            "latency_ms": round(latency_ms, 2),
            "flops": round(effective_flops, 1),
            "cost_usd": max(1e-7, round(cost_usd, 7)),
        }
