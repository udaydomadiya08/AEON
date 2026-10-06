"""
AEON Phase 9 — Standardized Baselines.

Defines standardized, immutable baselines (B0 to B10) representing alternative
computation and optimization strategies.
"""

from typing import Dict, List
from aeon.benchmarks.types import BaselineConfig


def get_canonical_baselines() -> List[BaselineConfig]:
    """Return the 11 canonical benchmark baselines (B0 through B10)."""
    return [
        BaselineConfig(
            id="B0_direct",
            name="Baseline 0 — Direct",
            description="Fixed direct strategy: 7B model, no memory, no tools, no speculation, no dynamic reasoning, full depth, FP16.",
            settings={
                "policy": "rules",
                "model_override": "aeon-medium",
                "memory": False,
                "tools": False,
                "speculate": False,
                "reasoning": False,
                "adaptive_depth": False,
                "hardware_aware": False,
                "precision": "fp16"
            }
        ),
        BaselineConfig(
            id="B1_fixed_small",
            name="Baseline 1 — Fixed Small",
            description="Always routes to smallest capable model (1B parameter profile).",
            settings={
                "policy": "rules",
                "model_override": "aeon-small",
                "memory": False,
                "tools": False,
                "speculate": False,
                "reasoning": False,
                "adaptive_depth": False,
                "hardware_aware": False,
                "precision": "fp16"
            }
        ),
        BaselineConfig(
            id="B2_fixed_medium",
            name="Baseline 2 — Fixed Medium",
            description="Always routes to medium model (7B parameter profile).",
            settings={
                "policy": "rules",
                "model_override": "aeon-medium",
                "memory": False,
                "tools": False,
                "speculate": False,
                "reasoning": False,
                "adaptive_depth": False,
                "hardware_aware": False,
                "precision": "fp16"
            }
        ),
        BaselineConfig(
            id="B3_fixed_large",
            name="Baseline 3 — Fixed Large",
            description="Always routes to largest model (70B parameter profile).",
            settings={
                "policy": "rules",
                "model_override": "aeon-large",
                "memory": False,
                "tools": False,
                "speculate": False,
                "reasoning": False,
                "adaptive_depth": False,
                "hardware_aware": False,
                "precision": "fp16"
            }
        ),
        BaselineConfig(
            id="B4_tool_first",
            name="Baseline 4 — Tool First",
            description="Always attempts deterministic tools before falling back to model.",
            settings={
                "policy": "rules",
                "memory": False,
                "tools": True,
                "speculate": False,
                "reasoning": False,
                "adaptive_depth": False,
                "hardware_aware": False,
                "precision": "fp16"
            }
        ),
        BaselineConfig(
            id="B5_fixed_heavy_reasoning",
            name="Baseline 5 — Fixed Heavy Reasoning",
            description="Always invokes heavy reasoning exploration and high reasoning tokens.",
            settings={
                "policy": "rules",
                "model_override": "aeon-medium",
                "memory": False,
                "tools": False,
                "speculate": False,
                "reasoning": True,
                "reasoning_strategy": "deep",
                "adaptive_depth": False,
                "hardware_aware": False,
                "precision": "fp16"
            }
        ),
        BaselineConfig(
            id="B6_fixed_full_depth",
            name="Baseline 6 — Fixed Full Depth",
            description="Standard AEON but with adaptive depth disabled (all 32 layers executed).",
            settings={
                "policy": "voi",
                "memory": True,
                "tools": True,
                "speculate": True,
                "reasoning": True,
                "adaptive_depth": False,
                "hardware_aware": True
            }
        ),
        BaselineConfig(
            id="B7_fixed_fp16",
            name="Baseline 7 — Fixed FP16",
            description="AEON with fixed FP16 precision (no adaptive precision or INT8 quantization).",
            settings={
                "policy": "voi",
                "memory": True,
                "tools": True,
                "speculate": True,
                "reasoning": True,
                "adaptive_depth": True,
                "hardware_aware": True,
                "precision": "fp16",
                "quantization": "none"
            }
        ),
        BaselineConfig(
            id="B8_fixed_int8",
            name="Baseline 8 — Fixed INT8",
            description="AEON with fixed INT8 quantization across all workloads.",
            settings={
                "policy": "voi",
                "memory": True,
                "tools": True,
                "speculate": True,
                "reasoning": True,
                "adaptive_depth": True,
                "hardware_aware": True,
                "precision": "int8",
                "quantization": "int8"
            }
        ),
        BaselineConfig(
            id="B9_phase7_aeon",
            name="Baseline 9 — Phase-7 AEON",
            description="AEON Phase 7 state: tools, memory, speculation, reasoning, and adaptive depth, but hardware/precision awareness OFF.",
            settings={
                "policy": "voi",
                "memory": True,
                "tools": True,
                "speculate": True,
                "reasoning": True,
                "adaptive_depth": True,
                "hardware_aware": False
            }
        ),
        BaselineConfig(
            id="B10_full_aeon",
            name="Baseline 10 — Full AEON",
            description="Full AEON with all Phase 1–8 optimizations enabled.",
            settings={
                "policy": "voi",
                "memory": True,
                "tools": True,
                "speculate": True,
                "reasoning": True,
                "adaptive_depth": True,
                "hardware_aware": True
            }
        )
    ]
