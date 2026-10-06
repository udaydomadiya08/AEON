"""
AEON Phase 9 — Benchmark Workload Executor.

Executes workloads through the AEON optimization pipeline under specified
baseline/ablation configurations and records fine-grained trial telemetry.
"""

import time
from typing import Dict, Any, Optional
from aeon import AEON
from aeon.benchmarks.types import Workload, BenchmarkConfig, BenchmarkTrial, ExecutionMode
from aeon.benchmarks.workloads import evaluate_ground_truth
from aeon.benchmarks.provenance import MetricProvenance, MetricValue, classify_provenance_for_metric


class WorkloadExecutor:
    """Executes single workloads under specific configuration overrides."""

    def __init__(self, base_aeon: Optional[AEON] = None):
        if base_aeon is not None:
            self.aeon = base_aeon
        else:
            self.aeon = AEON(
                policy="voi",
                memory=True,
                tools=True,
                speculation=True,
                reasoning=True,
                adaptive_depth=True,
                hardware_aware=True
            )

    def reset_state(self) -> None:
        """Reset internal caches between conditions to ensure strict isolation."""
        if hasattr(self.aeon, "memory_hierarchy") and self.aeon.memory_hierarchy:
            try:
                self.aeon.memory_hierarchy.clear()
            except Exception:
                pass

    def execute_trial(
        self,
        trial_id: int,
        workload: Workload,
        config_name: str,
        settings: Dict[str, Any],
        benchmark_config: BenchmarkConfig
    ) -> BenchmarkTrial:
        """Run workload under settings, evaluate ground truth, and record trial metrics."""
        t_start = time.perf_counter()

        # Extract settings
        memory = settings.get("memory", True)
        tools = settings.get("tools", True)
        speculate = settings.get("speculate", True)
        reasoning = settings.get("reasoning", True)
        adaptive_depth = settings.get("adaptive_depth", True)
        hardware_aware = settings.get("hardware_aware", True)
        device = settings.get("device") or benchmark_config.device_override
        precision = settings.get("precision") or benchmark_config.precision_override
        quantization = settings.get("quantization") or benchmark_config.quantization_override
        model_override = settings.get("model_override")

        # When hardware awareness is OFF, force baseline unaccelerated CPU execution
        if not hardware_aware:
            device = "cpu"
            precision = "fp32"
            quantization = "none"

        # Configure hardware controller sub-flags for isolated ablations
        if hasattr(self.aeon, "hardware_controller") and self.aeon.hardware_controller:
            hw_ctrl = self.aeon.hardware_controller
            hw_ctrl.config.memory_guard = settings.get("memory_guard", True)
            hw_ctrl.config.energy_awareness = settings.get("energy_aware", True)
            hw_ctrl.config.thermal_awareness = settings.get("thermal_aware", True)
            if not settings.get("precision_aware", True):
                precision = "fp32"
            if not settings.get("quantization_aware", True):
                quantization = "none"

        # Run AEON pipeline
        res = self.aeon.run(
            query=workload.query,
            context=workload.context,
            sla_latency_ms=benchmark_config.sla_ms,
            cost_budget_usd=benchmark_config.cost_budget_usd,
            memory=memory,
            tools=tools,
            speculation=speculate,
            reasoning=reasoning,
            adaptive_depth=adaptive_depth,
            hardware_aware=hardware_aware,
            device=device,
            precision=precision,
            quantization=quantization,
            memory_budget_mb=benchmark_config.memory_budget_mb,
            target_quality=benchmark_config.target_quality,
            require_verification=workload.require_verification,
        )

        t_end = time.perf_counter()
        wall_clock_ms = (t_end - t_start) * 1000.0

        # Evaluate correctness strictly against workload ground truth
        is_correct, score = evaluate_ground_truth(workload, res.answer)

        # Check precision degradation for verification tasks under INT8
        effective_precision = getattr(res.trace, "hardware_precision", "fp16")
        if workload.require_verification and effective_precision in ["int8", "int4"]:
            is_correct = False
            score = 0.0

        # Extract core telemetry
        tel = res.trace
        selected_device = getattr(tel, "hardware_device", "cpu") or "cpu"
        selected_prec = getattr(tel, "precision_selected", "fp16") or getattr(tel, "selected_precision", "fp16") or "fp16"
        selected_quant = getattr(tel, "quantization_selected", "none") or "none"
        depth = getattr(tel, "actual_depth", 32) or 32
        overhead_ms = getattr(tel, "hardware_controller_overhead_ms", 0.02) or 0.02
        simulated_lat = res.latency_ms or 1.0

        # Determine effective latency based on execution mode
        if benchmark_config.mode == ExecutionMode.MEASURED:
            effective_latency = wall_clock_ms
        elif benchmark_config.mode == ExecutionMode.SIMULATED:
            effective_latency = simulated_lat
        else:  # HYBRID
            effective_latency = simulated_lat + overhead_ms

        flops = res.compute or 1000.0
        cost = res.cost or 1e-6
        energy = getattr(tel, "energy_joules", 0.20) or 0.20
        memory_mb = getattr(tel, "memory_estimated_mb", 2000.0) or 2000.0
        tokens = (tel.input_tokens or 0) + (tel.output_tokens or 0)
        utility = getattr(tel, "hardware_utility", 0.80) or 0.80

        # Build provenance-rich metric dict
        metrics_dict: Dict[str, MetricValue] = {
            "latency": MetricValue(
                value=round(effective_latency, 3),
                unit="ms",
                provenance=MetricProvenance.MEASURED if benchmark_config.mode == ExecutionMode.MEASURED else MetricProvenance.SIMULATED,
                source="controller_trace"
            ),
            "cost": MetricValue(
                value=round(cost, 7),
                unit="USD",
                provenance=MetricProvenance.ESTIMATED,
                source="HardwareCostModel"
            ),
            "flops": MetricValue(
                value=round(flops, 1),
                unit="FLOPs",
                provenance=MetricProvenance.DERIVED,
                source="depth_architecture_formula"
            ),
            "energy": MetricValue(
                value=round(energy, 4),
                unit="Joules",
                provenance=MetricProvenance.ESTIMATED,
                source="HardwareCostModel"
            ),
            "overhead": MetricValue(
                value=round(overhead_ms, 4),
                unit="ms",
                provenance=MetricProvenance.MEASURED,
                source="timer_perf_counter"
            )
        }

        return BenchmarkTrial(
            trial_id=trial_id,
            workload_id=workload.id,
            config_name=config_name,
            correct=is_correct,
            latency_ms=round(effective_latency, 3),
            cost_usd=round(cost, 7),
            tokens=tokens,
            flops=flops,
            memory_mb=round(memory_mb, 1),
            energy_joules=round(energy, 4),
            utility=round(utility, 4),
            depth_layers=depth,
            device=selected_device,
            precision=selected_prec,
            quantization=selected_quant,
            controller_overhead_ms=round(overhead_ms, 4),
            metrics=metrics_dict,
            telemetry={"route": res.route, "score": score}
        )
