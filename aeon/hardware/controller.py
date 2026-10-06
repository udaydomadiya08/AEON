"""
AEON Phase 8: Hardware Controller & Joint Optimization Engine.

Performs joint candidate generation across (Device × Precision × Quantization × Model),
evaluates hard constraints (memory preflight, backend support, SLA, budget),
computes multi-objective economic utility, and resolves deterministic decisions with provenance.
"""

import time
import uuid
import math
from typing import Dict, List, Optional, Tuple, Any

from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    QuantizationMethod,
    CapabilityStatus,
    MemoryPressure,
    ExecutionMode,
    HardwareProfile,
    HardwareCandidate,
    HardwareDecision,
    HardwareConfig,
    ThermalState,
)
from aeon.hardware.detector import HardwareDetector
from aeon.hardware.precision import PrecisionManager
from aeon.hardware.quantization import QuantizationManager
from aeon.hardware.capabilities import HardwareCapabilityMatrix
from aeon.hardware.memory import HardwareMemoryManager
from aeon.hardware.cost_model import HardwareCostModel
from aeon.hardware.energy import EnergyEstimator
from aeon.hardware.thermal import ThermalMonitor
from aeon.hardware.backend import SimulationHardwareBackend


class HardwareController:
    """
    Central orchestrator for hardware-aware precision, quantization,
    memory preflighting, and economic compute decisions.
    """

    def __init__(
        self,
        config: Optional[HardwareConfig] = None,
        detector: Optional[HardwareDetector] = None,
        backend: Optional[SimulationHardwareBackend] = None,
    ):
        self.config = config or HardwareConfig()
        self.detector = detector or HardwareDetector()
        self.profile = self.detector.detect_hardware()

        self.precision_manager = PrecisionManager()
        self.quantization_manager = QuantizationManager()
        self.capability_matrix = HardwareCapabilityMatrix(self.profile)
        self.memory_manager = HardwareMemoryManager(safety_margin_ratio=self.config.memory_safety_margin)
        self.cost_model = HardwareCostModel()
        self.energy_estimator = EnergyEstimator()
        self.thermal_monitor = ThermalMonitor()
        self.backend = backend or SimulationHardwareBackend()

    def evaluate_and_select(
        self,
        query: str,
        model_name: str = "aeon-medium-7b",
        difficulty: float = 0.5,
        uncertainty: float = 0.5,
        sla_latency_ms: Optional[float] = None,
        cost_budget_usd: Optional[float] = None,
        memory_budget_mb: Optional[float] = None,
        target_quality: Optional[float] = None,
        preferred_device: Optional[str] = None,
        preferred_precision: Optional[str] = None,
        preferred_quantization: Optional[str] = None,
        depth_layers: Optional[int] = None,
        total_layers: Optional[int] = None,
    ) -> HardwareDecision:
        """
        Evaluate candidate hardware/precision configurations and select the
        economically optimal configuration that satisfies hard constraints.
        """
        start_time = time.perf_counter()
        decision_id = f"hwd-{uuid.uuid4().hex[:8]}"
        fallbacks: List[str] = []

        # Refresh dynamic system memory & thermal state
        self.profile = self.detector.refresh_dynamic(self.profile)
        self.capability_matrix.set_profile(self.profile)

        tokens_count = max(5, int(len(query) / 4.0))
        layer_ratio = 1.0
        if depth_layers and total_layers and total_layers > 0:
            layer_ratio = max(0.10, min(1.0, depth_layers / float(total_layers)))

        # 1. Generate Candidates
        candidates = self._generate_candidates(
            model_name=model_name,
            difficulty=difficulty,
            uncertainty=uncertainty,
            tokens_count=tokens_count,
            layer_ratio=layer_ratio,
            sla_latency_ms=sla_latency_ms,
            cost_budget_usd=cost_budget_usd,
            memory_budget_mb=memory_budget_mb,
            target_quality=target_quality or self.config.target_quality,
            preferred_device=preferred_device or self.config.preferred_device,
            preferred_precision=preferred_precision or self.config.preferred_precision,
            preferred_quantization=preferred_quantization or self.config.preferred_quantization,
            fallbacks=fallbacks,
            depth_layers=depth_layers,
        )

        # 2. Select Optimal Candidate
        viable_candidates = [c for c in candidates if c.is_viable]
        if viable_candidates:
            # Sort descending by utility
            viable_candidates.sort(key=lambda c: c.utility, reverse=True)
            chosen = viable_candidates[0]
            reason = "economic_utility_optimal"
        else:
            # Safe Fallback Candidate
            fits_mem = memory_budget_mb is None or memory_budget_mb > 0
            chosen = self._create_safe_fallback_candidate(
                model_name=model_name,
                tokens_count=tokens_count,
                layer_ratio=layer_ratio,
                difficulty=difficulty,
                preferred_device=preferred_device or self.config.preferred_device,
                preferred_precision=preferred_precision or self.config.preferred_precision,
                preferred_quantization=preferred_quantization or self.config.preferred_quantization,
                fits_memory=fits_mem,
            )
            fallbacks.append("all_candidates_rejected_reverting_to_safe_fallback")
            reason = "safe_fallback_under_constraint_pressure"

        overhead_ms = (time.perf_counter() - start_time) * 1000.0

        decision = HardwareDecision(
            decision_id=decision_id,
            device=chosen.device,
            precision=chosen.precision,
            quantization=chosen.quantization,
            model_name=chosen.model_name,
            memory_mb=chosen.estimated_memory_mb,
            latency_ms=chosen.estimated_latency_ms,
            cost_usd=chosen.estimated_cost_usd,
            energy_joules=chosen.estimated_energy_joules,
            quality_estimate=chosen.estimated_quality,
            utility=round(chosen.utility, 4),
            flops=chosen.estimated_flops,
            execution_mode=ExecutionMode.MEASURED if not self.profile.is_simulated else ExecutionMode.SIMULATED,
            candidates_evaluated=len(candidates),
            candidates=candidates,
            selected_candidate=chosen,
            fallbacks_triggered=fallbacks,
            decision_reason=reason,
            hardware_profile_id=self.profile.profile_id,
            hardware_profile=self.profile,
            provenance={
                "controller_overhead_ms": round(overhead_ms, 3),
                "device": chosen.device.value,
                "precision": chosen.precision.value,
                "quantization": chosen.quantization.value,
                "memory_pressure": chosen.memory_pressure.value,
                "unified_memory": self.profile.unified_memory,
            },
        )
        return decision

    def _generate_candidates(
        self,
        model_name: str,
        difficulty: float,
        uncertainty: float,
        tokens_count: int,
        layer_ratio: float,
        sla_latency_ms: Optional[float],
        cost_budget_usd: Optional[float],
        memory_budget_mb: Optional[float],
        target_quality: float,
        preferred_device: str,
        preferred_precision: str,
        preferred_quantization: str,
        fallbacks: List[str],
        depth_layers: Optional[int] = None,
    ) -> List[HardwareCandidate]:
        candidates: List[HardwareCandidate] = []

        # Device pool
        devices_to_eval = [self.profile.accelerator_type]
        if self.profile.accelerator_type != DeviceType.CPU:
            devices_to_eval.append(DeviceType.CPU)

        # Handle user preferred device override
        if preferred_device and preferred_device != "auto":
            try:
                forced_dev = DeviceType(preferred_device.lower())
                if forced_dev == self.profile.accelerator_type or forced_dev == DeviceType.CPU or self.config.allow_simulation:
                    devices_to_eval = [forced_dev]
                else:
                    fallbacks.append(f"requested_device_{preferred_device}_unsupported")
            except Exception:
                fallbacks.append(f"invalid_device_{preferred_device}")

        # Precision pool
        precisions_to_eval = [
            PrecisionType.FP16,
            PrecisionType.FP32,
            PrecisionType.BF16,
            PrecisionType.INT8,
        ]
        if self.config.allow_simulation or self.profile.cuda_available:
            precisions_to_eval.append(PrecisionType.FP8)

        if preferred_precision and preferred_precision != "auto":
            try:
                forced_prec = PrecisionType(preferred_precision.upper())
                precisions_to_eval = [forced_prec]
            except Exception:
                fallbacks.append(f"invalid_precision_{preferred_precision}")

        # Quantization pool
        quant_methods = [QuantizationMethod.NONE, QuantizationMethod.INT8_WEIGHT_ONLY]
        if preferred_quantization and preferred_quantization != "auto":
            try:
                forced_q = QuantizationMethod(preferred_quantization.lower())
                quant_methods = [forced_q]
            except Exception:
                fallbacks.append(f"invalid_quantization_{preferred_quantization}")

        base_quality = 0.92 if "7b" in model_name else (0.97 if "70b" in model_name else 0.86)
        thermal_risk = self.thermal_monitor.evaluate_thermal_risk(
            ThermalState(pressure=self.profile.thermal_pressure)
        )

        for dev in devices_to_eval:
            for prec in precisions_to_eval:
                for quant in quant_methods:
                    cand_id = f"cand-{dev.value}-{prec.value}-{quant.value}"

                    # 1. Check capability
                    cap_status, cap_reason = self.capability_matrix.evaluate_compatibility(
                        device=dev,
                        model_name=model_name,
                        precision=prec,
                    )
                    if cap_status == CapabilityStatus.UNSUPPORTED and not self.config.allow_simulation:
                        candidates.append(
                            self._create_rejected_candidate(
                                cand_id, model_name, dev, prec, quant, f"unsupported_capability: {cap_reason}"
                            )
                        )
                        continue

                    # 2. Check memory preflight
                    mem_assessment = self.memory_manager.preflight_check(
                        model_name=model_name,
                        precision=prec,
                        device=dev,
                        profile=self.profile,
                        quantization=quant,
                        memory_budget_mb=memory_budget_mb,
                    )
                    if self.config.memory_guard and not mem_assessment.fits_in_memory:
                        candidates.append(
                            self._create_rejected_candidate(
                                cand_id,
                                model_name,
                                dev,
                                prec,
                                quant,
                                f"oom_risk: requires {mem_assessment.estimated_model_mb}MB but only {mem_assessment.available_memory_mb}MB available",
                            )
                        )
                        continue

                    # 3. Compute cost metrics
                    metrics = self.cost_model.compute_metrics(
                        model_name=model_name,
                        device=dev,
                        precision=prec,
                        quantization=quant,
                        tokens_count=tokens_count,
                        layer_ratio=layer_ratio,
                    )
                    latency_ms = metrics["latency_ms"]
                    cost_usd = metrics["cost_usd"]
                    flops = metrics["flops"]

                    # 4. Check budgets
                    if sla_latency_ms is not None and latency_ms > sla_latency_ms:
                        candidates.append(
                            self._create_rejected_candidate(
                                cand_id, model_name, dev, prec, quant, f"sla_exceeded: {latency_ms}ms > {sla_latency_ms}ms"
                            )
                        )
                        continue

                    if cost_budget_usd is not None and cost_usd > cost_budget_usd:
                        candidates.append(
                            self._create_rejected_candidate(
                                cand_id, model_name, dev, prec, quant, f"budget_exceeded: ${cost_usd} > ${cost_budget_usd}"
                            )
                        )
                        continue

                    # 5. Compute quality impact
                    est_quality = self.precision_manager.estimate_quality_impact(
                        precision=prec,
                        base_quality=base_quality,
                        difficulty=difficulty,
                        model_name=model_name,
                    )
                    if quant != QuantizationMethod.NONE:
                        quant_res = self.quantization_manager.evaluate_quantization(
                            config=self.quantization_manager.create_config(method=quant, target_precision=prec),
                            model_name=model_name,
                            device=dev,
                            difficulty=difficulty,
                        )
                        est_quality = max(0.10, est_quality + quant_res.quality_impact)

                    if est_quality < (target_quality - 0.05):
                        candidates.append(
                            self._create_rejected_candidate(
                                cand_id, model_name, dev, prec, quant, f"quality_below_target: {est_quality:.2f} < {target_quality:.2f}"
                            )
                        )
                        continue

                    # 6. Compute energy
                    energy_est = self.energy_estimator.estimate_energy(
                        device=dev,
                        precision=prec,
                        latency_ms=latency_ms,
                        tokens_count=tokens_count,
                    )

                    # 7. Multi-Objective Economic Utility
                    # Utility = Quality - Costs/Penalties (Normalized)
                    norm_lat = min(2.0, latency_ms / 1000.0)
                    norm_cost = min(2.0, cost_usd * 1000.0)
                    norm_energy = min(2.0, energy_est.energy_joules / 100.0)
                    norm_mem = min(1.0, mem_assessment.estimated_model_mb / max(1.0, mem_assessment.total_memory_mb))
                    prec_risk = 0.05 if prec in [PrecisionType.INT8, PrecisionType.FP8] else 0.0

                    energy_penalty = (self.config.energy_lambda * norm_energy) if self.config.energy_awareness else 0.0
                    thermal_penalty = (self.config.thermal_lambda * thermal_risk) if self.config.thermal_awareness else 0.0
                    mem_penalty = (self.config.memory_lambda * norm_mem) if self.config.memory_guard else 0.0

                    utility = (
                        est_quality
                        - (self.config.latency_lambda * norm_lat)
                        - (self.config.cost_lambda * norm_cost)
                        - energy_penalty
                        - mem_penalty
                        - thermal_penalty
                        - (self.config.precision_risk_lambda * prec_risk)
                    )

                    candidates.append(
                        HardwareCandidate(
                            candidate_id=cand_id,
                            model_name=model_name,
                            device=dev,
                            precision=prec,
                            quantization=quant,
                            depth=depth_layers,
                            mod_capacity=1.0,
                            estimated_quality=round(est_quality, 4),
                            estimated_latency_ms=latency_ms,
                            estimated_cost_usd=cost_usd,
                            estimated_memory_mb=mem_assessment.estimated_model_mb,
                            estimated_energy_joules=energy_est.energy_joules,
                            estimated_flops=flops,
                            memory_pressure=mem_assessment.memory_pressure,
                            thermal_risk=thermal_risk,
                            precision_risk=prec_risk,
                            utility=round(utility, 4),
                            is_viable=True,
                        )
                    )

        return candidates

    def _create_rejected_candidate(
        self,
        candidate_id: str,
        model_name: str,
        device: DeviceType,
        precision: PrecisionType,
        quantization: QuantizationMethod,
        reason: str,
    ) -> HardwareCandidate:
        return HardwareCandidate(
            candidate_id=candidate_id,
            model_name=model_name,
            device=device,
            precision=precision,
            quantization=quantization,
            estimated_quality=0.0,
            estimated_latency_ms=99999.0,
            estimated_cost_usd=999.0,
            estimated_memory_mb=0.0,
            estimated_energy_joules=0.0,
            estimated_flops=0.0,
            memory_pressure=MemoryPressure.CRITICAL,
            thermal_risk=1.0,
            precision_risk=1.0,
            utility=-999.0,
            is_viable=False,
            rejection_reason=reason,
        )

    def _create_safe_fallback_candidate(
        self,
        model_name: str,
        tokens_count: int,
        layer_ratio: float,
        difficulty: float,
        preferred_device: Optional[Any] = None,
        preferred_precision: Optional[Any] = None,
        preferred_quantization: Optional[Any] = None,
        fits_memory: bool = True,
    ) -> HardwareCandidate:
        dev = preferred_device or DeviceType.CPU
        if isinstance(dev, str):
            try:
                dev = DeviceType(dev)
            except Exception:
                dev = DeviceType.CPU
        prec = preferred_precision or PrecisionType.FP32
        if isinstance(prec, str):
            try:
                prec = PrecisionType(prec)
            except Exception:
                prec = PrecisionType.FP32
        quant = preferred_quantization or QuantizationMethod.NONE
        if isinstance(quant, str):
            try:
                quant = QuantizationMethod(quant)
            except Exception:
                quant = QuantizationMethod.NONE
        if prec == PrecisionType.FP8 or (dev == DeviceType.MPS and prec == PrecisionType.FP8):
            prec = PrecisionType.FP32

        metrics = self.cost_model.compute_metrics(
            model_name=model_name,
            device=dev,
            precision=prec,
            quantization=quant,
            tokens_count=tokens_count,
            layer_ratio=layer_ratio,
        )
        mem_mb = self.memory_manager.estimate_model_memory_mb(model_name, prec, quant)
        thermal_risk = self.thermal_monitor.evaluate_thermal_risk(
            ThermalState(pressure=self.profile.thermal_pressure)
        )
        norm_lat = min(2.0, metrics["latency_ms"] / 1000.0)
        norm_cost = min(2.0, metrics["cost_usd"] * 1000.0)
        energy_est = self.energy_estimator.estimate_energy(
            device=dev, precision=prec, latency_ms=metrics["latency_ms"], tokens_count=tokens_count
        )
        norm_energy = min(2.0, energy_est.energy_joules / 100.0)
        norm_mem = min(1.0, mem_mb / max(1.0, self.profile.ram_total_mb))
        prec_risk = 0.05 if prec in [PrecisionType.INT8, PrecisionType.FP8] else 0.0

        energy_penalty = (self.config.energy_lambda * norm_energy) if self.config.energy_awareness else 0.0
        thermal_penalty = (self.config.thermal_lambda * thermal_risk) if self.config.thermal_awareness else 0.0
        mem_penalty = (self.config.memory_lambda * norm_mem) if self.config.memory_guard else 0.0

        fallback_utility = (
            0.80
            - (self.config.latency_lambda * norm_lat)
            - (self.config.cost_lambda * norm_cost)
            - energy_penalty
            - mem_penalty
            - thermal_penalty
            - (self.config.precision_risk_lambda * prec_risk)
        )

        return HardwareCandidate(
            candidate_id=f"fallback-{dev.value}-{prec.value}",
            model_name=model_name,
            device=dev,
            precision=prec,
            quantization=quant,
            estimated_quality=0.80,
            estimated_latency_ms=metrics["latency_ms"],
            estimated_cost_usd=metrics["cost_usd"],
            estimated_memory_mb=mem_mb,
            estimated_energy_joules=energy_est.energy_joules,
            estimated_flops=metrics["flops"],
            memory_pressure=MemoryPressure.NORMAL,
            thermal_risk=thermal_risk,
            precision_risk=prec_risk,
            utility=round(fallback_utility, 4),
            is_viable=True,
            fits_memory=fits_memory,
        )


global_hardware_controller = HardwareController()
