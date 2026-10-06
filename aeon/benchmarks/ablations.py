"""
AEON Phase 9 — Canonical 10-Way Ablation Matrix & Strict Isolation Verifier.

Defines the canonical 10-way ablations (A0 to A9) and labeled extensions (A10 to A12).
Verifies that each ablation disables precisely one declared parameter.
"""

from typing import Dict, List, Any, Tuple
from aeon.benchmarks.types import AblationConfig


class AblationIsolationError(ValueError):
    """Raised when an ablation changes more or fewer parameters than declared."""
    pass


BASE_FULL_AEON_SETTINGS: Dict[str, Any] = {
    "policy": "voi",
    "memory": True,
    "tools": True,
    "speculate": True,
    "reasoning": True,
    "adaptive_depth": True,
    "hardware_aware": True,
    "precision_aware": True,
    "quantization_aware": True,
    "memory_guard": True,
    "energy_aware": True,
    "thermal_aware": True,
    "tool_routing": True,
    "dynamic_stopping": True,
    "memory_hierarchy": True,
}


def get_canonical_ablations() -> List[AblationConfig]:
    """Return the canonical 10-way ablation matrix (A0 through A9)."""
    ablations = [
        AblationConfig(
            id="A0_full_aeon",
            name="Full AEON",
            description="All Phase 1–8 capabilities active.",
            target_component="none",
            changed_parameters={},
            unchanged_parameters=dict(BASE_FULL_AEON_SETTINGS)
        ),
        AblationConfig(
            id="A1_hardware_off",
            name="Hardware OFF",
            description="Hardware-aware device selection disabled (fixed unaccelerated CPU baseline).",
            target_component="hardware_aware",
            changed_parameters={"hardware_aware": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "hardware_aware"}
        ),
        AblationConfig(
            id="A2_precision_off",
            name="Precision OFF",
            description="Precision adaptation disabled (forces baseline unquantized FP32).",
            target_component="precision_aware",
            changed_parameters={"precision_aware": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "precision_aware"}
        ),
        AblationConfig(
            id="A3_quantization_off",
            name="Quantization OFF",
            description="Integer quantization disabled (INT8/INT4 candidates suppressed).",
            target_component="quantization_aware",
            changed_parameters={"quantization_aware": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "quantization_aware"}
        ),
        AblationConfig(
            id="A4_memory_guard_off",
            name="Memory Guard OFF",
            description="Preflight memory headroom checks disabled.",
            target_component="memory_guard",
            changed_parameters={"memory_guard": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "memory_guard"}
        ),
        AblationConfig(
            id="A5_energy_off",
            name="Energy OFF",
            description="Energy consumption penalties zeroed in candidate ranking.",
            target_component="energy_aware",
            changed_parameters={"energy_aware": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "energy_aware"}
        ),
        AblationConfig(
            id="A6_thermal_off",
            name="Thermal OFF",
            description="Thermal throttling penalties zeroed in candidate ranking.",
            target_component="thermal_aware",
            changed_parameters={"thermal_aware": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "thermal_aware"}
        ),
        AblationConfig(
            id="A7_depth_off",
            name="Depth OFF",
            description="Phase 7 adaptive depth disabled (all layers executed).",
            target_component="adaptive_depth",
            changed_parameters={"adaptive_depth": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "adaptive_depth"}
        ),
        AblationConfig(
            id="A8_reasoning_off",
            name="Reasoning OFF",
            description="Phase 6 adaptive reasoning controller disabled.",
            target_component="reasoning",
            changed_parameters={"reasoning": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "reasoning"}
        ),
        AblationConfig(
            id="A9_speculation_off",
            name="Speculation OFF",
            description="Phase 5 speculative decoding disabled.",
            target_component="speculate",
            changed_parameters={"speculate": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "speculate"}
        )
    ]
    return ablations


def get_extended_ablations() -> List[AblationConfig]:
    """Return labeled extended ablations (A10 through A12)."""
    return [
        AblationConfig(
            id="A10_memory_hierarchy_off",
            name="Memory Hierarchy OFF",
            description="Phase 3 multi-tier memory hierarchy (L1-L4) disabled.",
            target_component="memory",
            changed_parameters={"memory": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "memory"}
        ),
        AblationConfig(
            id="A11_tool_routing_off",
            name="Tool Routing OFF",
            description="Phase 4 deterministic tool routing disabled.",
            target_component="tools",
            changed_parameters={"tools": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "tools"}
        ),
        AblationConfig(
            id="A12_dynamic_stopping_off",
            name="Dynamic Stopping OFF",
            description="Phase 2 VOI dynamic stopping disabled (runs full iteration budget).",
            target_component="dynamic_stopping",
            changed_parameters={"dynamic_stopping": False},
            unchanged_parameters={k: v for k, v in BASE_FULL_AEON_SETTINGS.items() if k != "dynamic_stopping"}
        )
    ]


def verify_ablation_isolation(ablation: AblationConfig, effective_settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Formally verify that effective_settings differs from BASE_FULL_AEON_SETTINGS
    strictly by the declared changed_parameters.
    """
    actual_diff = {}
    actual_unchanged = {}

    for k, base_val in BASE_FULL_AEON_SETTINGS.items():
        curr_val = effective_settings.get(k, base_val)
        if curr_val != base_val:
            actual_diff[k] = (base_val, curr_val)
        else:
            actual_unchanged[k] = curr_val

    # Ensure all declared changed params are in actual_diff
    for k, expected_val in ablation.changed_parameters.items():
        if k not in actual_diff or actual_diff[k][1] != expected_val:
            raise AblationIsolationError(
                f"Ablation {ablation.id} failed isolation: expected parameter {k}={expected_val}, "
                f"but got diff {actual_diff.get(k)}"
            )

    # Ensure no undeclared params changed
    unexpected = [k for k in actual_diff if k not in ablation.changed_parameters]
    if unexpected:
        raise AblationIsolationError(
            f"Ablation {ablation.id} failed isolation: unexpected parameter change in {unexpected}!"
        )

    return actual_diff, actual_unchanged
