"""
[INPUT]: numpy, scipy.stats
[OUTPUT]: statistical_tests, validator
[POS]: experiments/cross_validation/ module initialization
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from experiments.cross_validation.statistical_tests import (
    shapiro_wilk_test,
    paired_t_test,
    wilcoxon_test,
    compare_fold_results,
    StatisticalTestResult
)

from experiments.cross_validation.validator import (
    KFoldCrossValidator,
    FoldSplit,
    DatasetProtocol
)

__all__ = [
    "shapiro_wilk_test",
    "paired_t_test",
    "wilcoxon_test",
    "compare_fold_results",
    "StatisticalTestResult",
    "KFoldCrossValidator",
    "FoldSplit",
    "DatasetProtocol"
]
