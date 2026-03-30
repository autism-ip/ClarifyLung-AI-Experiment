"""
[INPUT]: scipy.stats, numpy
[OUTPUT]: Statistical test functions
[POS]: experiments/cross_validation/ statistical significance testing
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticalTestResult:
    """Statistical test result container"""
    statistic: float
    p_value: float
    significant: bool
    test_name: str


def shapiro_wilk_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Shapiro-Wilk normality test

    [INPUT]:  numpy array, samples > 3
    [OUTPUT]: (statistic, p_value)
    """
    if len(data) < 3:
        raise ValueError("Need at least 3 samples for Shapiro-Wilk test")
    return stats.shapiro(data)


def paired_t_test(baseline: np.ndarray, variant: np.ndarray) -> Tuple[float, float]:
    """
    Paired t-test for comparing two related samples

    [INPUT]: baseline and variant arrays of equal length
    [OUTPUT]: (statistic, p_value)
    """
    if len(baseline) != len(variant):
        raise ValueError("Baseline and variant must have same length")
    if len(baseline) < 2:
        raise ValueError("Need at least 2 pairs for paired t-test")
    return stats.ttest_rel(baseline, variant)


def wilcoxon_test(baseline: np.ndarray, variant: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for non-parametric comparison

    [INPUT]: baseline and variant arrays of equal length
    [OUTPUT]: (statistic, p_value)
    """
    if len(baseline) != len(variant):
        raise ValueError("Baseline and variant must have same length")
    if len(baseline) < 5:
        raise ValueError("Need at least 5 pairs for Wilcoxon test")
    return stats.wilcoxon(baseline, variant)


def compare_fold_results(
    baseline_scores: List[float],
    variant_scores: List[float],
    alpha: float = 0.05
) -> dict:
    """
    Compare cross-validation results between baseline and variant

    [INPUT]: List of fold-wise scores for both models, significance level
    [OUTPUT]: dict with p_value, significant, test_name, statistic
    """
    baseline_arr = np.array(baseline_scores)
    variant_arr = np.array(variant_scores)

    # Use paired t-test for fold results
    if len(baseline_arr) == len(variant_arr) and len(baseline_arr) >= 5:
        statistic, p_value = paired_t_test(baseline_arr, variant_arr)
        test_name = "paired_t_test"
    else:
        # Fall back to Wilcoxon for smaller samples
        statistic, p_value = wilcoxon_test(baseline_arr, variant_arr)
        test_name = "wilcoxon_test"

    return {
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "test_name": test_name,
        "statistic": float(statistic)
    }
