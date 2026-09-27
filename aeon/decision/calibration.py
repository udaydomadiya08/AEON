"""
Calibration and Empirical Telemetry Learning Layer for AEON.

Measures calibration error (ECE, Brier Score) and maintains empirical historical
accuracy and resource profiles to eliminate overconfident routing.
Conforms to AEON Specification Sections 11 and 12.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field


class CalibrationMetrics(BaseModel):
    num_samples: int
    brier_score: float
    expected_calibration_error: float  # ECE in [0.0, 1.0]
    bin_confidences: List[float] = Field(default_factory=list)
    bin_accuracies: List[float] = Field(default_factory=list)
    bin_counts: List[int] = Field(default_factory=list)


class CalibrationTracker:
    """
    Maintains predictions and outcomes to compute calibration metrics (ECE, Brier Score)
    and detect model overconfidence / underconfidence.
    """

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self._predictions: List[float] = []
        self._outcomes: List[int] = []

    def record(self, confidence: float, correct: bool) -> None:
        self._predictions.append(float(np.clip(confidence, 0.0, 1.0)))
        self._outcomes.append(1 if correct else 0)

    def compute_metrics(self) -> CalibrationMetrics:
        n = len(self._predictions)
        if n == 0:
            return CalibrationMetrics(
                num_samples=0,
                brier_score=0.0,
                expected_calibration_error=0.0,
            )

        preds = np.array(self._predictions)
        targets = np.array(self._outcomes)

        # 1. Brier score: mean squared difference between confidence and binary outcome
        brier_score = float(np.mean((preds - targets) ** 2))

        # 2. Expected Calibration Error (ECE)
        bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        ece = 0.0

        for i in range(self.num_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            if i == self.num_bins - 1:
                mask = (preds >= bin_start) & (preds <= bin_end)
            else:
                mask = (preds >= bin_start) & (preds < bin_end)

            count = int(np.sum(mask))
            bin_counts.append(count)

            if count > 0:
                bin_conf = float(np.mean(preds[mask]))
                bin_acc = float(np.mean(targets[mask]))
                bin_confidences.append(round(bin_conf, 4))
                bin_accuracies.append(round(bin_acc, 4))
                ece += (count / n) * abs(bin_acc - bin_conf)
            else:
                bin_confidences.append(round((bin_start + bin_end) / 2.0, 4))
                bin_accuracies.append(0.0)

        return CalibrationMetrics(
            num_samples=n,
            brier_score=round(brier_score, 4),
            expected_calibration_error=round(float(ece), 4),
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
        )


class EmpiricalCalibrationStore:
    """
    Stores empirical historical observations from telemetry to calibrate future
    quality estimates and prevent overconfidence.
    """

    def __init__(self):
        # Key: (model_name, task_type) -> list of (difficulty, confidence, correct, latency_ms, cost_usd)
        self._history: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def record_run(
        self,
        model_name: str,
        task_type: str,
        difficulty: float,
        confidence: float,
        correct: bool,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        key = (model_name, task_type)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append({
            "difficulty": difficulty,
            "confidence": confidence,
            "correct": correct,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        })

    def get_calibrated_success_prob(
        self,
        model_name: str,
        task_type: str,
        difficulty: float,
        prior: float,
    ) -> float:
        """Returns empirical success probability if enough samples exist, else blends with prior."""
        key = (model_name, task_type)
        records = self._history.get(key, [])
        if not records:
            return prior

        # Filter to similar difficulties (within ±0.20)
        similar = [r for r in records if abs(r["difficulty"] - difficulty) <= 0.20]
        if len(similar) < 3:
            return prior

        empirical_p = np.mean([1 if r["correct"] else 0 for r in similar])
        # Bayesian shrinkage toward prior
        weight = min(1.0, len(similar) / 20.0)
        return float(round((1 - weight) * prior + weight * empirical_p, 4))


# Global calibration tracker instance
global_calibration_tracker = CalibrationTracker()
global_calibration_store = EmpiricalCalibrationStore()
