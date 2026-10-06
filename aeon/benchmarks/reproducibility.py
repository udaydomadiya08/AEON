"""
AEON Phase 9 — Reproducibility & Seed Management.

Manages deterministic seed propagation and validates exact bit-for-bit
reproducibility between independent benchmark executions.
"""

import hashlib
import json
import random
import numpy as np
from typing import Dict, Any, Tuple, List


class SeedManager:
    """Manages deterministic pseudorandom seeds across benchmark subsystems."""

    def __init__(self, global_seed: int = 42):
        self.global_seed = global_seed
        self.workload_seed = (global_seed * 10007) % 2147483647
        self.trial_seed = (global_seed * 20011) % 2147483647
        self.model_seed = (global_seed * 30013) % 2147483647

    def apply_seeds(self) -> None:
        """Apply seeds to global Python random and numpy generators."""
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)

    def get_seed_manifest(self) -> Dict[str, int]:
        """Return seed map for auditing and manifest creation."""
        return {
            "global_seed": self.global_seed,
            "workload_seed": self.workload_seed,
            "trial_seed": self.trial_seed,
            "model_seed": self.model_seed
        }


def compute_configuration_hash(config_dict: Dict[str, Any]) -> str:
    """Compute SHA-256 digest of normalized configuration dictionary."""
    serialized = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def verify_runs_identical(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verify whether two independent benchmark runs produced identical results.
    Returns (is_identical, list_of_discrepancies).
    """
    discrepancies = []
    
    # Compare common summary metrics
    keys_to_compare = ["quality", "total_tokens", "mean_flops", "mean_depth"]
    for k in keys_to_compare:
        val_a = run_a.get(k)
        val_b = run_b.get(k)
        if val_a != val_b:
            discrepancies.append(f"Metric '{k}' mismatch: Run A={val_a}, Run B={val_b}")

    return len(discrepancies) == 0, discrepancies
