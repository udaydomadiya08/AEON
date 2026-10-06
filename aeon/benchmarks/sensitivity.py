"""
AEON Phase 9 — Sensitivity & Decision Stability Engine.

Performs multidimensional parameter sweeps and quantifies decision stability
under micro-perturbations.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class SensitivityAnalysisEngine:
    """Executes parameter sweeps over controller utility weights and constraints."""

    @staticmethod
    def sweep_latency_weight(
        controller_evaluator,
        candidate_data_list: List[Dict[str, Any]],
        weight_range: List[float] = None
    ) -> List[Dict[str, Any]]:
        """Sweep lambda_latency from 0.05 to 0.70 and track winning candidate index."""
        if weight_range is None:
            weight_range = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65]

        results = []
        for w_lat in weight_range:
            best_idx = -1
            best_utility = -float("inf")
            for idx, cand in enumerate(candidate_data_list):
                u = controller_evaluator(cand, {"latency": w_lat, "cost": 0.15, "quality": 0.40})
                if u > best_utility:
                    best_utility = u
                    best_idx = idx

            winning = candidate_data_list[best_idx]
            results.append({
                "weight_latency": w_lat,
                "selected_candidate_index": best_idx,
                "selected_device": winning.get("device"),
                "selected_precision": winning.get("precision"),
                "utility": best_utility
            })
        return results

    @staticmethod
    def measure_decision_stability(
        decision_function,
        base_request: Dict[str, Any],
        perturbation_std: float = 0.02,
        trials: int = 20
    ) -> Dict[str, float]:
        """
        Measure action, model, and precision flip rates under input perturbations.
        A perfectly stable router will have 0.0 flip rate for small perturbations.
        """
        base_decision = decision_function(base_request)
        base_model = base_decision.get("model")
        base_precision = base_decision.get("precision")
        base_depth = base_decision.get("depth")

        model_flips = 0
        precision_flips = 0
        depth_flips = 0

        for _ in range(trials):
            perturbed = dict(base_request)
            if "sla_ms" in perturbed:
                noise = np.random.normal(0, perturbation_std * perturbed["sla_ms"])
                perturbed["sla_ms"] = max(10.0, perturbed["sla_ms"] + noise)
            
            dec = decision_function(perturbed)
            if dec.get("model") != base_model:
                model_flips += 1
            if dec.get("precision") != base_precision:
                precision_flips += 1
            if dec.get("depth") != base_depth:
                depth_flips += 1

        return {
            "model_flip_rate": round(model_flips / trials, 4),
            "precision_flip_rate": round(precision_flips / trials, 4),
            "depth_flip_rate": round(depth_flips / trials, 4),
            "action_flip_rate": round(max(model_flips, precision_flips, depth_flips) / trials, 4),
            "trials": trials
        }
