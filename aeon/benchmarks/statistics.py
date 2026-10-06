"""
AEON Phase 9 — Statistical Analysis Engine.

Calculates descriptive and inferential statistics over benchmark repetitions:
mean, median, standard deviation, percentiles (p50, p90, p95, p99),
and 95% Student's-t confidence intervals.
"""

import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from aeon.benchmarks.types import StatisticalSummary


# Standard two-tailed Student's t critical values for 95% confidence level (alpha=0.05)
T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045, 30: 2.042
}


def get_t_critical_95(df: int) -> float:
    """Return two-tailed critical t-value for 95% CI."""
    if df <= 0:
        return 1.96
    if df in T_TABLE_95:
        return T_TABLE_95[df]
    if df > 30:
        return 1.960
    # Nearest key lookup
    closest = min(T_TABLE_95.keys(), key=lambda k: abs(k - df))
    return T_TABLE_95[closest]


def compute_statistical_summary(values: List[float]) -> StatisticalSummary:
    """Compute complete statistical summary for a numeric metric series."""
    n = len(values)
    if n == 0:
        return StatisticalSummary(
            mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0,
            p50=0.0, p90=0.0, p95=0.0, p99=0.0, sample_size=0
        )

    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p50_val = float(np.percentile(arr, 50))
    p90_val = float(np.percentile(arr, 90))
    p95_val = float(np.percentile(arr, 95))
    p99_val = float(np.percentile(arr, 99))

    ci_lower = None
    ci_upper = None
    if n > 1 and std_val > 1e-12:
        df = n - 1
        t_crit = get_t_critical_95(df)
        margin = t_crit * (std_val / math.sqrt(n))
        ci_lower = mean_val - margin
        ci_upper = mean_val + margin

    return StatisticalSummary(
        mean=round(mean_val, 6),
        median=round(median_val, 6),
        std=round(std_val, 6),
        min=round(min_val, 6),
        max=round(max_val, 6),
        p50=round(p50_val, 6),
        p90=round(p90_val, 6),
        p95=round(p95_val, 6),
        p99=round(p99_val, 6),
        ci_95_lower=round(ci_lower, 6) if ci_lower is not None else None,
        ci_95_upper=round(ci_upper, 6) if ci_upper is not None else None,
        sample_size=n
    )


def compute_paired_comparison(candidate_vals: List[float], baseline_vals: List[float]) -> Dict[str, Any]:
    """Perform paired comparison between candidate and baseline series on identical workloads."""
    if len(candidate_vals) != len(baseline_vals):
        raise ValueError("Paired comparison requires identical sample lengths")
    n = len(candidate_vals)
    if n == 0:
        return {"mean_diff": 0.0, "p_value_approx": 1.0, "significant": False}

    diffs = np.array(candidate_vals) - np.array(baseline_vals)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    
    if n > 1 and std_diff > 1e-12:
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        df = n - 1
        t_crit = get_t_critical_95(df)
        significant = abs(t_stat) >= t_crit
    elif abs(mean_diff) > 1e-9:
        t_stat = 999.0 if mean_diff > 0 else -999.0
        significant = True
    else:
        t_stat = 0.0
        significant = False

    return {
        "mean_diff": round(mean_diff, 6),
        "std_diff": round(std_diff, 6),
        "t_statistic": round(t_stat, 4),
        "significant": significant,
        "sample_size": n
    }
