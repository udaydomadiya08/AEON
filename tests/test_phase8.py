"""
Comprehensive Test Suite for AEON Phase 8: Hardware & Quantization.

Covers:
1. Hardware Detection & Profiling (Apple Silicon, CPU, mocked CUDA, missing tools, privacy)
2. Precision System (Profiles, compatibility, fallbacks, task sensitivity)
3. Quantization Subsystem (Simulated vs Real, quality degradation, analytical sizing)
4. Memory Management & OOM Preflight (Unified vs discrete, safety margins, rejection)
5. Energy & Thermal Modeling (Power x time estimation, thermal risk penalties)
6. Hardware-Aware Cost & Utility Economics (Dimensional safety, NaN/Inf bounds, penalty terms)
7. Hardware Controller (Candidate generation, multi-objective ranking, overrides, deterministic replay)
8. End-to-End AEON System Integration (Phases 1-7 interaction, dynamic stopping, telemetry)
9. Adversarial & Edge Cases (Zero/negative budgets, invalid metrics, missing backends)
10. CLI & Telemetry Verification
"""

import math
import pytest
from unittest.mock import patch, MagicMock

from aeon.hardware.types import (
    DeviceType,
    PrecisionType,
    QuantizationMethod,
    CapabilityStatus,
    ThermalPressure,
    MemoryPressure,
    HardwareProfile,
    PrecisionProfile,
    QuantizationConfig,
    QuantizationResult,
    HardwareConfig,
    HardwareCandidate,
    HardwareDecision,
    ThermalState,
)
from aeon.hardware.detector import HardwareDetector
from aeon.hardware.profiles import (
    APPLE_SILICON_M_PROFILE,
    NVIDIA_A100_PROFILE,
    NVIDIA_H100_PROFILE,
    CPU_SERVER_PROFILE,
    estimate_model_memory_mb,
)
from aeon.hardware.capabilities import HardwareCapabilityMatrix
from aeon.hardware.precision import PrecisionManager, STANDARD_PRECISION_PROFILES
from aeon.hardware.quantization import QuantizationManager
from aeon.hardware.memory import HardwareMemoryManager
from aeon.hardware.cost_model import HardwareCostModel
from aeon.hardware.energy import EnergyEstimator
from aeon.hardware.thermal import ThermalMonitor
from aeon.hardware.controller import HardwareController
from aeon.hardware.telemetry import extract_hardware_telemetry
from click.testing import CliRunner
from aeon import AEON, InferenceRequest, TaskType, DifficultyLevel
from aeon.cli.main import cli


# ==============================================================================
# 1. HARDWARE DETECTION & PROFILING
# ==============================================================================

class TestHardwareDetector:
    def test_detect_current_host(self):
        detector = HardwareDetector()
        profile = detector.detect_hardware(force_refresh=True)

        assert isinstance(profile, HardwareProfile)
        assert profile.profile_id != ""
        assert profile.machine_arch in ["arm64", "x86_64", "aarch64", "amd64", "unknown", "arm64e"]
        assert profile.cpu_cores_logical >= 1
        assert profile.ram_total_mb > 0
        assert profile.ram_available_mb >= 0
        assert profile.ram_available_mb <= profile.ram_total_mb

    def test_privacy_guarantee(self):
        detector = HardwareDetector()
        profile = detector.detect_hardware()

        # Telemetry extracted from profile must not contain sensitive strings
        profile_dict = profile.model_dump()
        profile_str = str(profile_dict).lower()
        for forbidden in ["serial", "mac_address", "uday", "username"]:
            assert forbidden not in profile_str, f"Found sensitive leak: {forbidden}"

    def test_cpu_fallback_detection(self):
        detector = HardwareDetector()
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            profile = detector.detect_hardware(force_refresh=True)
            assert profile.accelerator_type == DeviceType.CPU
            assert profile.cuda_available is False
            assert profile.mps_available is False

    def test_mocked_cuda_detection(self):
        detector = HardwareDetector()
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-80GB"), \
             patch("torch.cuda.get_device_capability", return_value=(8, 0)), \
             patch("torch.cuda.get_device_properties") as mock_prop, \
             patch("torch.cuda.memory_allocated", return_value=10 * 1024**3), \
             patch("torch.cuda.memory_reserved", return_value=12 * 1024**3):
            mock_prop.return_value.total_memory = 80 * 1024**3
            profile = detector.detect_hardware(force_refresh=True)

            assert profile.accelerator_type == DeviceType.CUDA
            assert profile.cuda_available is True
            assert profile.cuda_device_count == 2
            assert profile.cuda_compute_capability == "8.0"
            assert profile.accelerator_memory_mb == 80 * 1024
            assert profile.unified_memory is False

    def test_static_reference_profiles(self):
        assert APPLE_SILICON_M_PROFILE.accelerator_type == DeviceType.MPS
        assert APPLE_SILICON_M_PROFILE.unified_memory is True
        assert NVIDIA_A100_PROFILE.accelerator_type == DeviceType.CUDA
        assert NVIDIA_A100_PROFILE.accelerator_memory_mb == 81920.0
        assert CPU_SERVER_PROFILE.accelerator_type == DeviceType.CPU


# ==============================================================================
# 2. PRECISION SYSTEM & COMPATIBILITY
# ==============================================================================

class TestPrecisionSystem:
    def test_precision_profiles_integrity(self):
        manager = PrecisionManager()
        for ptype in [PrecisionType.FP32, PrecisionType.FP16, PrecisionType.BF16, PrecisionType.INT8, PrecisionType.FP8]:
            prof = manager.get_profile(ptype)
            assert prof.precision == ptype
            assert prof.bits > 0
            assert prof.memory_factor > 0.0
            assert prof.compute_factor > 0.0
            assert -0.10 <= prof.quality_delta <= 0.05

    def test_fallback_chain_resolution(self):
        manager = PrecisionManager()
        # Profile supporting only FP32 and FP16:
        supported = {
            "fp32": CapabilityStatus.SUPPORTED.value,
            "fp16": CapabilityStatus.SUPPORTED.value,
            "fp8": CapabilityStatus.UNSUPPORTED.value,
        }
        resolved, fallbacks, reason = manager.resolve_fallback(
            PrecisionType.FP8, DeviceType.MPS, supported
        )
        assert resolved == PrecisionType.FP16
        assert len(fallbacks) > 0

    def test_task_complexity_quality_impact(self):
        manager = PrecisionManager()
        # High difficulty tasks should show lower net quality retention
        q_simple = manager.estimate_quality_impact(PrecisionType.INT8, base_quality=0.90, difficulty=0.1)
        q_complex = manager.estimate_quality_impact(PrecisionType.INT8, base_quality=0.90, difficulty=0.9)
        assert q_complex < q_simple


# ==============================================================================
# 3. QUANTIZATION SUBSYSTEM
# ==============================================================================

class TestQuantizationSubsystem:
    def test_analytical_memory_reduction(self):
        qm = QuantizationManager()
        cfg = qm.create_config(
            method=QuantizationMethod.INT8_DYNAMIC,
            source_precision=PrecisionType.FP16,
            target_precision=PrecisionType.INT8,
            calibrated=True,
        )
        res = qm.evaluate_quantization(cfg, "aeon-medium-7b", DeviceType.MPS, difficulty=0.5)

        assert res.memory_reduction_factor == 0.50
        assert res.quality_impact < 0.0
        assert res.is_simulated is True

    def test_calibrated_vs_uncalibrated_quality_retention(self):
        qm = QuantizationManager()
        cfg_calibrated = qm.create_config(
            method=QuantizationMethod.INT8_DYNAMIC,
            calibrated=True,
        )
        cfg_uncalibrated = qm.create_config(
            method=QuantizationMethod.INT8_DYNAMIC,
            calibrated=False,
        )
        res_cal = qm.evaluate_quantization(cfg_calibrated, "aeon-medium-7b", DeviceType.MPS)
        res_uncal = qm.evaluate_quantization(cfg_uncalibrated, "aeon-medium-7b", DeviceType.MPS)
        # Calibrated should have less severe negative quality impact
        assert abs(res_cal.quality_impact) < abs(res_uncal.quality_impact)


# ==============================================================================
# 4. MEMORY MANAGEMENT & OOM PREFLIGHT
# ==============================================================================

class TestMemoryManager:
    def test_preflight_sufficient_memory(self):
        mm = HardwareMemoryManager(safety_margin_ratio=0.10)
        profile = APPLE_SILICON_M_PROFILE
        assessment = mm.preflight_check(
            model_name="aeon-small-1.5b",
            precision=PrecisionType.FP16,
            device=DeviceType.MPS,
            profile=profile,
        )
        assert assessment.fits is True
        assert assessment.oom_risk < 0.5
        assert assessment.recommended_fallback is None

    def test_preflight_insufficient_memory_oom_protection(self):
        mm = HardwareMemoryManager(safety_margin_ratio=0.10)
        profile = CPU_SERVER_PROFILE
        # Model with massive parameters on small budget
        assessment = mm.preflight_check(
            model_name="aeon-large-70b",
            precision=PrecisionType.FP32,
            device=DeviceType.CPU,
            profile=profile,
            memory_budget_mb=1000.0,  # 1 GB budget for a 70B model
        )
        assert assessment.fits is False
        assert assessment.oom_risk >= 0.9
        assert assessment.recommended_fallback is not None

    def test_memory_budget_constraint_enforcement(self):
        mm = HardwareMemoryManager(safety_margin_ratio=0.10)
        profile = APPLE_SILICON_M_PROFILE
        assessment = mm.preflight_check(
            model_name="aeon-medium-7b",
            precision=PrecisionType.FP16,
            device=DeviceType.MPS,
            profile=profile,
            memory_budget_mb=500.0,
        )
        assert assessment.fits is False


# ==============================================================================
# 5. ENERGY & THERMAL MODELING
# ==============================================================================

class TestEnergyAndThermal:
    def test_energy_estimation_dimensional_safety(self):
        ee = EnergyEstimator()
        est = ee.estimate_energy(
            device=DeviceType.MPS,
            precision=PrecisionType.FP16,
            latency_ms=250.0,
            tokens_count=50,
        )
        assert est.joules > 0.0
        assert est.watts > 0.0
        assert math.isfinite(est.joules)
        assert est.measurement_status.value.upper() in ["ESTIMATED", "SIMULATED"]

    def test_thermal_penalty_scaling(self):
        tm = ThermalMonitor()
        nominal_state = ThermalState(pressure=ThermalPressure.NOMINAL)
        nominal_penalty = tm.evaluate_thermal_risk(nominal_state)
        assert nominal_penalty == 0.0

        heavy_state = ThermalState(pressure=ThermalPressure.HEAVY, is_throttling=True)
        heavy_penalty = tm.evaluate_thermal_risk(heavy_state)
        assert heavy_penalty > 0.5
        assert tm.recommend_thermal_mitigation(heavy_state) is not None


# ==============================================================================
# 6. HARDWARE-AWARE COST & UTILITY ECONOMICS
# ==============================================================================

class TestHardwareEconomics:
    def test_cost_model_bounds_and_safety(self):
        cm = HardwareCostModel()
        metrics = cm.compute_metrics(
            model_name="aeon-medium-7b",
            device=DeviceType.CUDA,
            precision=PrecisionType.FP16,
            quantization=QuantizationMethod.NONE,
            tokens_count=100,
        )
        assert metrics["latency_ms"] > 0.0
        assert metrics["cost_usd"] > 0.0
        assert metrics["flops"] > 0.0
        assert math.isfinite(metrics["latency_ms"])
        assert math.isfinite(metrics["cost_usd"])

    def test_numerical_safety_zero_tokens(self):
        cm = HardwareCostModel()
        metrics = cm.compute_metrics(
            model_name="aeon-medium-7b",
            device=DeviceType.CPU,
            precision=PrecisionType.FP32,
            quantization=QuantizationMethod.NONE,
            tokens_count=0,
        )
        assert metrics["latency_ms"] >= 0.0
        assert math.isfinite(metrics["latency_ms"])


# ==============================================================================
# 7. HARDWARE CONTROLLER
# ==============================================================================

class TestHardwareController:
    def test_candidate_generation_and_selection(self):
        controller = HardwareController(HardwareConfig(enabled=True))
        decision = controller.evaluate_and_select(
            query="Analyze computational efficiency across precision levels",
            model_name="aeon-medium-7b",
            difficulty=0.6,
        )
        assert isinstance(decision, HardwareDecision)
        assert decision.selected is not None
        assert decision.selected.precision in [PrecisionType.FP16, PrecisionType.FP32, PrecisionType.INT8, PrecisionType.BF16]
        assert decision.selected.utility > -1e9
        assert len(decision.candidates) > 0
        assert decision.hardware_profile.profile_id != ""

    def test_user_override_enforcement(self):
        controller = HardwareController(HardwareConfig(enabled=True))
        decision = controller.evaluate_and_select(
            query="Simple calculation",
            model_name="aeon-small-1.5b",
            preferred_precision="fp32",
        )
        assert decision.selected.precision == PrecisionType.FP32

    def test_deterministic_decisions(self):
        controller = HardwareController(HardwareConfig(enabled=True))
        d1 = controller.evaluate_and_select("Deterministic query test", "aeon-medium-7b", difficulty=0.5)
        d2 = controller.evaluate_and_select("Deterministic query test", "aeon-medium-7b", difficulty=0.5)

        assert d1.selected.device == d2.selected.device
        assert d1.selected.precision == d2.selected.precision
        assert d1.selected.quantization == d2.selected.quantization
        assert pytest.approx(d1.selected.utility, rel=1e-5) == d2.selected.utility


# ==============================================================================
# 8. INTEGRATION WITH PHASES 1-7
# ==============================================================================

class TestPhaseIntegrations:
    def test_aeon_run_hardware_aware(self):
        system = AEON(
            hardware_aware=True,
            preferred_device="auto",
        )
        assert system.hardware is not None
        resp = system.run("Explain dynamic quantization tradeoffs")
        assert resp.response != ""
        assert resp.telemetry is not None
        assert resp.telemetry.hardware_profile_id != ""
        assert resp.telemetry.selected_precision != ""

    def test_aeon_user_precision_override(self):
        system = AEON(hardware_aware=True)
        resp = system.run("Test precision override", precision="fp32")
        assert resp.telemetry.selected_precision.lower() == "fp32"

    def test_hardware_cost_modulates_reasoning_and_depth(self):
        controller = HardwareController(HardwareConfig(enabled=True))
        # High difficulty task with compact model that fits in available memory
        d_mps = controller.evaluate_and_select(
            query="Deep mathematical proof",
            model_name="aeon-decision-jev",
            difficulty=0.9,
            preferred_device="mps",
        )
        d_cpu = controller.evaluate_and_select(
            query="Deep mathematical proof",
            model_name="aeon-decision-jev",
            difficulty=0.9,
            preferred_device="cpu",
        )
        assert d_mps.selected is not None
        assert d_cpu.selected is not None
        assert d_mps.selected.device == DeviceType.MPS
        assert d_cpu.selected.device == DeviceType.CPU
        assert d_mps.selected.estimated_latency_ms <= d_cpu.selected.estimated_latency_ms


# ==============================================================================
# 9. ADVERSARIAL & EDGE CASES
# ==============================================================================

class TestAdversarialAndEdgeCases:
    def test_zero_memory_budget_fallback(self):
        controller = HardwareController(HardwareConfig(enabled=True))
        decision = controller.evaluate_and_select(
            query="Zero memory test",
            model_name="aeon-large-70b",
            memory_budget_mb=0.0,
        )
        assert decision.selected is not None
        assert decision.selected.fits_memory is False

    def test_unsupported_precision_fallback(self):
        pm = PrecisionManager()
        supported = {
            "fp32": CapabilityStatus.SUPPORTED.value,
            "fp8": CapabilityStatus.UNSUPPORTED.value,
        }
        resolved, fallbacks, reason = pm.resolve_fallback(PrecisionType.FP8, DeviceType.CPU, supported)
        assert resolved in [PrecisionType.BF16, PrecisionType.FP16, PrecisionType.FP32]
        assert len(fallbacks) > 0

    def test_nan_cost_protection(self):
        cm = HardwareCostModel()
        metrics = cm.compute_metrics(
            model_name="aeon-medium-7b",
            device=DeviceType.MPS,
            precision=PrecisionType.FP16,
            quantization=QuantizationMethod.NONE,
            tokens_count=10,
        )
        assert not math.isnan(metrics["latency_ms"])
        assert not math.isnan(metrics["cost_usd"])


# ==============================================================================
# 10. CLI COMMANDS VERIFICATION
# ==============================================================================

class TestCliHardware:
    def test_cli_hardware_info(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "info"])
        assert result.exit_code == 0
        assert "AEON HARDWARE PROFILE" in result.output

    def test_cli_hardware_detect(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "detect"])
        assert result.exit_code == 0
        assert "profile_id" in result.output
        assert "machine_arch" in result.output

    def test_cli_hardware_precision(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "precision"])
        assert result.exit_code == 0
        assert "FP32" in result.output
        assert "FP16" in result.output
        assert "INT8" in result.output

    def test_cli_hardware_quantization(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "quantization"])
        assert result.exit_code == 0
        assert "AEON QUANTIZATION METHODS" in result.output
        assert "SIMULATED" in result.output

    def test_cli_hardware_config(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "config"])
        assert result.exit_code == 0
        assert "enabled" in result.output
        assert "auto_detect" in result.output

    def test_cli_hardware_break_even(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "break-even"])
        assert result.exit_code == 0
        assert "break_even_tokens_for_overhead" in result.output
        assert "break_even_quality_loss_tolerance" in result.output

    def test_cli_hardware_overhead(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "overhead"])
        assert result.exit_code == 0
        assert "mean_ms" in result.output
        assert "p95_ms" in result.output

    def test_cli_hardware_benchmark(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "benchmark", "--runs", "1"])
        assert result.exit_code == 0
        assert "Baseline A" in result.output

    def test_cli_hardware_ablation(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["hardware", "ablation"])
        assert result.exit_code == 0
        assert "Full AEON Phase 8" in result.output


# ==============================================================================
# 11. FORMAL BEHAVIORAL BENCHMARK SUITE (AUDIT SECTION 35)
# ==============================================================================

class TestPhase8BehavioralBenchmarkSuite:
    """
    Explicit ground-truth behavioural test suite validating Section 35:
    Task A: Quality-sensitive -> FP32/FP16 preferred over INT8
    Task B: Memory-constrained -> INT8 selected under memory cap
    Task C: Large model + high-memory CUDA profile -> viable accelerator candidate
    Task D: CPU-only environment -> deterministic CPU fallback
    Task E: Unsupported FP8 -> deterministic fallback to supported precision
    Task F: Thermal pressure -> elevated thermal penalty shifts utility
    Task G: Quantization quality violation -> quantization rejected
    Task H: Deterministic arithmetic -> Phase 4 tool remains preferred
    """

    def test_task_a_quality_sensitive_prefers_high_precision(self):
        ctrl = HardwareController()
        # High quality target 0.98 cannot tolerate INT8 precision degradation
        decision = ctrl.evaluate_and_select(
            query="Formal mathematical proof of Godel incompleteness",
            model_name="aeon-medium-7b",
            target_quality=0.98,
        )
        assert decision.precision in [PrecisionType.FP32, PrecisionType.FP16, PrecisionType.BF16]
        assert decision.quantization == QuantizationMethod.NONE

    def test_task_b_memory_constrained_selects_quantization(self):
        ctrl = HardwareController()
        # 1.5B model in FP16 is ~3.2 GB, which exceeds 2.5 GB budget, but fits in INT8 weight quantization (~1.6 GB)
        decision = ctrl.evaluate_and_select(
            query="Analyze massive log corpus under tight memory constraints",
            model_name="aeon-small-1.5b",
            memory_budget_mb=2500.0,
            target_quality=0.75,
            preferred_precision="FP16",
        )
        assert decision.quantization in [QuantizationMethod.INT8_WEIGHT_ONLY, QuantizationMethod.INT8_DYNAMIC]

    def test_task_c_large_model_cuda_profile(self):
        ctrl = HardwareController()
        matrix = HardwareCapabilityMatrix(NVIDIA_A100_PROFILE)
        cand = ctrl.cost_model.compute_metrics(
            model_name="aeon-large-70b",
            device=DeviceType.CUDA,
            precision=PrecisionType.FP16,
            quantization=QuantizationMethod.NONE,
        )
        assert cand["latency_ms"] < 250.0
        assert cand["cost_usd"] > 0.0

    def test_task_d_cpu_only_fallback(self):
        detector = HardwareDetector()
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            profile = detector.detect_hardware(force_refresh=True)
            ctrl = HardwareController(detector=detector)
            decision = ctrl.evaluate_and_select(
                query="CPU-only test task",
                model_name="aeon-small-1.5b",
            )
            assert decision.device == DeviceType.CPU

    def test_task_e_unsupported_fp8_fallback(self):
        ctrl = HardwareController()
        matrix = HardwareCapabilityMatrix(ctrl.profile)
        mps_fp8 = matrix.check_support(DeviceType.MPS, PrecisionType.FP8)
        assert mps_fp8 == CapabilityStatus.UNSUPPORTED
        decision = ctrl.evaluate_and_select(
            query="Requesting unsupported FP8",
            model_name="aeon-medium-7b",
            preferred_precision="FP8",
        )
        assert decision.precision in [PrecisionType.FP16, PrecisionType.FP32, PrecisionType.BF16]

    def test_task_f_thermal_pressure_shifts_utility(self):
        ctrl_nominal = HardwareController()
        ctrl_critical = HardwareController()
        ctrl_critical.profile.thermal_pressure = ThermalPressure.CRITICAL
        ctrl_critical.profile.thermal_temperature_c = 95.0

        d_nom = ctrl_nominal.evaluate_and_select("compute task", model_name="aeon-small-1.5b")
        d_crit = ctrl_critical.evaluate_and_select("compute task", model_name="aeon-small-1.5b")
        assert d_crit.utility < d_nom.utility

    def test_task_g_quantization_quality_violation_rejected(self):
        ctrl = HardwareController()
        decision = ctrl.evaluate_and_select(
            query="Strict zero-tolerance precision task",
            model_name="aeon-medium-7b",
            target_quality=0.99,
        )
        for c in decision.candidates:
            if c.quantization != QuantizationMethod.NONE:
                assert not c.is_viable or "quality_below_target" in (c.rejection_reason or "")

    def test_task_h_deterministic_arithmetic_prefers_tool(self):
        aeon = AEON(policy="voi", tools=True, hardware_aware=True)
        req = InferenceRequest(
            query="What is 17.5% of 840?",
            task_hint="arithmetic",
            tools=True,
            hardware_aware=True,
        )
        res = aeon.run(req)
        assert res.route == "tool"
        assert res.trace.hardware_enabled is True


# ==============================================================================
# 12. ECONOMIC & ABLATION CAUSAL INTEGRITY (AUDIT SECTION 20)
# ==============================================================================

class TestPhase8EconomicAndAblationCausalIntegrity:
    """
    Formal causal integrity tests verifying:
    - Cost changes across materially different configurations
    - Quality/$ arithmetic validity
    - Memory Guard feasibility gating
    - Precision and Quantization ablation isolation
    - Energy & Thermal ablation penalty gating
    - Hardware profile economic divergence
    - Phase 7 depth scaling propagation
    - Quality constraint precision rejection
    - Controller selected optimum equals independent utility computation
    """

    def test_cost_changes_with_configuration(self):
        cm = HardwareCostModel()
        m_fp32 = cm.compute_metrics("aeon-medium-7b", DeviceType.MPS, PrecisionType.FP32, QuantizationMethod.NONE, tokens_count=100)
        m_fp16 = cm.compute_metrics("aeon-medium-7b", DeviceType.MPS, PrecisionType.FP16, QuantizationMethod.NONE, tokens_count=100)
        m_int8 = cm.compute_metrics("aeon-medium-7b", DeviceType.MPS, PrecisionType.INT8, QuantizationMethod.INT8_WEIGHT_ONLY, tokens_count=100)
        m_depth = cm.compute_metrics("aeon-medium-7b", DeviceType.MPS, PrecisionType.FP16, QuantizationMethod.NONE, tokens_count=100, layer_ratio=0.5)

        assert m_fp32["cost_usd"] > m_fp16["cost_usd"], f"FP32 cost ({m_fp32['cost_usd']}) must exceed FP16 ({m_fp16['cost_usd']})"
        assert m_fp16["cost_usd"] > m_int8["cost_usd"], f"FP16 cost ({m_fp16['cost_usd']}) must exceed INT8 ({m_int8['cost_usd']})"
        assert m_fp16["cost_usd"] > m_depth["cost_usd"], f"Full depth cost ({m_fp16['cost_usd']}) must exceed reduced depth ({m_depth['cost_usd']})"

    def test_quality_dollar_is_mathematically_correct(self):
        from aeon.benchmarks.metrics import calculate_aggregate_metrics
        from aeon.types import Telemetry
        traces = [
            Telemetry(
                request_id="t1", task_type="classification", route="small_model", model="aeon-small-1.5b",
                reasoning_mode="NONE", input_tokens=10, output_tokens=20, reasoning_tokens=0,
                latency_ms=10.0, compute_estimate=1e8, memory_mb=100.0, estimated_energy=0.01,
                estimated_cost=0.0001, cache_hit=False, speculative=False, verification=False,
                confidence=0.9, escalated=False, correct=True
            ),
            Telemetry(
                request_id="t2", task_type="classification", route="small_model", model="aeon-small-1.5b",
                reasoning_mode="NONE", input_tokens=10, output_tokens=20, reasoning_tokens=0,
                latency_ms=10.0, compute_estimate=1e8, memory_mb=100.0, estimated_energy=0.01,
                estimated_cost=0.0001, cache_hit=False, speculative=False, verification=False,
                confidence=0.9, escalated=False, correct=True
            )
        ]
        agg = calculate_aggregate_metrics("test", traces)
        expected_q_dollar = agg.accuracy / agg.total_cost_usd
        assert abs(agg.quality_per_dollar - expected_q_dollar) < 1e-4

    def test_memory_guard_changes_feasibility(self):
        ctrl_on = HardwareController(config=HardwareConfig(memory_guard=True, memory_safety_margin=0.15))
        ctrl_off = HardwareController(config=HardwareConfig(memory_guard=False, memory_safety_margin=0.0))

        dec_on = ctrl_on.evaluate_and_select("test query", model_name="aeon-medium-7b", memory_budget_mb=100.0)
        dec_off = ctrl_off.evaluate_and_select("test query", model_name="aeon-medium-7b", memory_budget_mb=100.0)

        viable_on = [c for c in dec_on.candidates if c.is_viable]
        viable_off = [c for c in dec_off.candidates if c.is_viable]
        assert len(viable_on) == 0, "Guard ON must reject all candidates under 100MB budget"
        assert len(viable_off) > 0, "Guard OFF must allow candidates even under 100MB budget"

    def test_precision_ablation_is_isolated(self):
        ctrl = HardwareController()
        # Forced FP32
        dec_fp32 = ctrl.evaluate_and_select("test", model_name="aeon-medium-7b", preferred_precision="FP32")
        assert dec_fp32.precision == PrecisionType.FP32

        # Forced FP16
        dec_fp16 = ctrl.evaluate_and_select("test", model_name="aeon-medium-7b", preferred_precision="FP16")
        assert dec_fp16.precision == PrecisionType.FP16

    def test_quantization_ablation_is_isolated(self):
        ctrl = HardwareController()
        # Quantization OFF: forces none
        dec_none = ctrl.evaluate_and_select("test", model_name="aeon-medium-7b", preferred_quantization="none")
        assert dec_none.quantization == QuantizationMethod.NONE

        # Quantization forced to INT8
        dec_int8 = ctrl.evaluate_and_select("test", model_name="aeon-medium-7b", preferred_quantization="int8_weight_only", target_quality=0.70)
        assert dec_int8.quantization == QuantizationMethod.INT8_WEIGHT_ONLY

    def test_energy_ablation_changes_penalty(self):
        ctrl_on = HardwareController(config=HardwareConfig(energy_awareness=True, energy_lambda=1.0))
        ctrl_off = HardwareController(config=HardwareConfig(energy_awareness=False, energy_lambda=1.0))

        d_on = ctrl_on.evaluate_and_select("test", model_name="aeon-small-1.5b")
        d_off = ctrl_off.evaluate_and_select("test", model_name="aeon-small-1.5b")

        c_on = [c for c in d_on.candidates if c.candidate_id == d_on.selected_candidate.candidate_id][0]
        c_off = [c for c in d_off.candidates if c.candidate_id == d_on.selected_candidate.candidate_id][0]
        assert c_off.utility > c_on.utility, "Utility without energy penalty must be strictly greater than with energy penalty"

    def test_thermal_ablation_changes_penalty(self):
        ctrl_on = HardwareController(config=HardwareConfig(thermal_awareness=True, thermal_lambda=2.0))
        ctrl_off = HardwareController(config=HardwareConfig(thermal_awareness=False, thermal_lambda=2.0))
        ctrl_on.profile.thermal_pressure = ThermalPressure.CRITICAL
        ctrl_off.profile.thermal_pressure = ThermalPressure.CRITICAL

        d_on = ctrl_on.evaluate_and_select("test", model_name="aeon-small-1.5b")
        d_off = ctrl_off.evaluate_and_select("test", model_name="aeon-small-1.5b")

        c_on = [c for c in d_on.candidates if c.candidate_id == d_on.selected_candidate.candidate_id][0]
        c_off = [c for c in d_off.candidates if c.candidate_id == d_on.selected_candidate.candidate_id][0]
        assert c_off.utility > c_on.utility, "Utility without thermal penalty must be strictly greater than with thermal penalty under critical heat"

    def test_hardware_profile_changes_economics(self):
        cm = HardwareCostModel()
        m_cpu = cm.compute_metrics("aeon-medium-7b", DeviceType.CPU, PrecisionType.FP32, QuantizationMethod.NONE)
        m_mps = cm.compute_metrics("aeon-medium-7b", DeviceType.MPS, PrecisionType.FP16, QuantizationMethod.NONE)
        m_cuda = cm.compute_metrics("aeon-medium-7b", DeviceType.CUDA, PrecisionType.FP16, QuantizationMethod.NONE)

        assert m_cpu["latency_ms"] > m_mps["latency_ms"]
        assert m_mps["latency_ms"] > m_cuda["latency_ms"]

    def test_phase7_depth_changes_compute(self):
        cm = HardwareCostModel()
        m_full = cm.compute_metrics("aeon-medium-7b", DeviceType.MPS, PrecisionType.FP16, QuantizationMethod.NONE, layer_ratio=1.0)
        m_half = cm.compute_metrics("aeon-medium-7b", DeviceType.MPS, PrecisionType.FP16, QuantizationMethod.NONE, layer_ratio=0.5)

        assert m_full["flops"] > m_half["flops"]
        assert m_full["latency_ms"] > m_half["latency_ms"]
        assert m_full["cost_usd"] > m_half["cost_usd"]

    def test_quality_constraint_rejects_bad_precision(self):
        ctrl = HardwareController()
        dec = ctrl.evaluate_and_select("proof", model_name="aeon-medium-7b", target_quality=0.98)
        assert dec.precision in [PrecisionType.FP32, PrecisionType.FP16, PrecisionType.BF16]
        assert dec.quantization == QuantizationMethod.NONE

    def test_controller_matches_independent_utility(self):
        ctrl = HardwareController()
        dec = ctrl.evaluate_and_select("test query", model_name="aeon-small-1.5b")
        chosen = dec.selected_candidate
        assert chosen is not None

        # Calculate utility independently
        norm_lat = min(2.0, chosen.estimated_latency_ms / 1000.0)
        norm_cost = min(2.0, chosen.estimated_cost_usd * 1000.0)
        norm_energy = min(2.0, chosen.estimated_energy_joules / 100.0)
        norm_mem = min(1.0, chosen.estimated_memory_mb / max(1.0, ctrl.profile.ram_total_mb))
        prec_risk = 0.05 if chosen.precision in [PrecisionType.INT8, PrecisionType.FP8] else 0.0

        energy_penalty = (ctrl.config.energy_lambda * norm_energy) if ctrl.config.energy_awareness else 0.0
        thermal_penalty = (ctrl.config.thermal_lambda * chosen.thermal_risk) if ctrl.config.thermal_awareness else 0.0
        mem_penalty = (ctrl.config.memory_lambda * norm_mem) if ctrl.config.memory_guard else 0.0

        indep_utility = (
            chosen.estimated_quality
            - (ctrl.config.latency_lambda * norm_lat)
            - (ctrl.config.cost_lambda * norm_cost)
            - energy_penalty
            - mem_penalty
            - thermal_penalty
            - (ctrl.config.precision_risk_lambda * prec_risk)
        )
        assert abs(round(indep_utility, 4) - chosen.utility) < 1e-4
