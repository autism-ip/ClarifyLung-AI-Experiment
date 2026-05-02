"""
统计检验模块（扩展版）
[INPUT]: 多组实验结果（不同模型或不同种子）
[OUTPUT]: paired t-test, ANOVA, p值报告, 效应量（Cohen's d）
[POS]: experiments/ 统计显著性分析核心组件
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md

功能：对实验结果进行统计显著性检验
要求：
  - paired t-test（配对t检验，比较两个模型）
  - ANOVA（方差分析，比较多组）
  - p值报告
  - 效应量（Cohen's d）
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy import stats


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class StatisticalTestResult:
    """统计检验结果容器"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'statistic': float(self.statistic),
            'p_value': float(self.p_value),
            'significant': bool(self.significant),
            'alpha': self.alpha,
            'effect_size': float(self.effect_size) if self.effect_size is not None else None,
            'effect_interpretation': self.effect_interpretation,
            'confidence_interval': self.confidence_interval,
            **self.additional_info,
        }

    def __str__(self) -> str:
        sig_mark = "***" if self.significant else "ns"
        ci_str = f", CI=[{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]" \
                 if self.confidence_interval else ""
        es_str = f", Cohen's d={self.effect_size:.3f} ({self.effect_interpretation})" \
                 if self.effect_size else ""
        return (f"{self.test_name}: statistic={self.statistic:.4f}, "
                f"p={self.p_value:.6f} [{sig_mark}]{ci_str}{es_str}")


# =============================================================================
# 效应量计算
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    计算Cohen's d效应量

    Cohen's d = (mean1 - mean2) / pooled_std

    解释标准：
        |d| < 0.2: 可忽略 (negligible)
        0.2 <= |d| < 0.5: 小 (small)
        0.5 <= |d| < 0.8: 中等 (medium)
        |d| >= 0.8: 大 (large)
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return float(d)


def interpret_cohens_d(d: float) -> str:
    """解释Cohen's d的大小"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    计算Hedges' g（Cohen's d的小样本校正版本）

    适用于样本量较小的情况（n < 20）
    """
    d = cohens_d(group1, group2)
    n1, n2 = len(group1), len(group2)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


# =============================================================================
# 正态性检验
# =============================================================================

def shapiro_wilk_test(data: np.ndarray) -> Tuple[float, float]:
    """
    Shapiro-Wilk正态性检验

    Returns:
        (statistic, p_value)
    """
    if len(data) < 3:
        raise ValueError("Need at least 3 samples for Shapiro-Wilk test")
    if len(data) > 5000:
        warnings.warn("Shapiro-Wilk test may be unreliable for n > 5000")

    return stats.shapiro(data)


def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    检查数据是否服从正态分布

    Returns:
        {'is_normal': bool, 'test': str, 'statistic': float, 'p_value': float}
    """
    if len(data) >= 3:
        stat, p = shapiro_wilk_test(data)
        is_normal = p > alpha
        return {
            'is_normal': is_normal,
            'test': 'shapiro_wilk',
            'statistic': float(stat),
            'p_value': float(p),
        }
    else:
        return {
            'is_normal': None,
            'test': 'none',
            'statistic': None,
            'p_value': None,
            'note': 'Sample size too small for normality test',
        }


# =============================================================================
# 配对t检验
# =============================================================================

def paired_t_test(
    baseline: np.ndarray,
    variant: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> StatisticalTestResult:
    """
    配对t检验

    用于比较同一组样本在两种不同条件下的差异
    （例如：同一数据集上两个不同模型的准确率，使用不同随机种子）

    Args:
        baseline: 基线模型结果数组 [n_seeds]
        variant: 对比模型结果数组 [n_seeds]
        alpha: 显著性水平
        alternative: 'two-sided', 'less', 'greater'

    Returns:
        StatisticalTestResult
    """
    baseline = np.asarray(baseline)
    variant = np.asarray(variant)

    if len(baseline) != len(variant):
        raise ValueError("Baseline and variant must have same length")
    if len(baseline) < 2:
        raise ValueError("Need at least 2 pairs for paired t-test")

    # 正态性检查
    diff = baseline - variant
    normality = check_normality(diff, alpha)

    # 执行检验
    statistic, p_value = stats.ttest_rel(baseline, variant, alternative=alternative)

    # 计算效应量（配对t检验使用差值的标准化均值）
    effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    effect_interp = interpret_cohens_d(effect_size)

    # 计算置信区间
    ci = None
    if len(diff) > 1:
        se = stats.sem(diff)
        df = len(diff) - 1
        t_crit = stats.t.ppf(1 - alpha/2, df)
        mean_diff = np.mean(diff)
        ci = (float(mean_diff - t_crit * se), float(mean_diff + t_crit * se))

    return StatisticalTestResult(
        test_name="paired_t_test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(effect_size),
        effect_interpretation=effect_interp,
        confidence_interval=ci,
        additional_info={'normality_check': normality}
    )


# =============================================================================
# 独立样本t检验
# =============================================================================

def independent_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,
    alternative: str = 'two-sided'
) -> StatisticalTestResult:
    """
    独立样本t检验（Welch's t-test）

    用于比较两组独立样本的差异
    （例如：两个不同数据集上的模型性能）

    Args:
        group1: 第一组样本
        group2: 第二组样本
        alpha: 显著性水平
        equal_var: 是否假设方差齐性
        alternative: 备择假设方向
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Need at least 2 samples per group")

    # Welch's t-test（不假设方差齐性，更稳健）
    statistic, p_value = stats.ttest_ind(
        group1, group2,
        equal_var=equal_var,
        alternative=alternative
    )

    # 效应量
    d = cohens_d(group1, group2)
    ci = None

    return StatisticalTestResult(
        test_name="welch_t_test" if not equal_var else "student_t_test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=d,
        effect_interpretation=interpret_cohens_d(d),
        confidence_interval=ci,
    )


# =============================================================================
# 非参数检验：Wilcoxon符号秩检验
# =============================================================================

def wilcoxon_signed_rank_test(
    baseline: np.ndarray,
    variant: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> StatisticalTestResult:
    """
    Wilcoxon符号秩检验（配对样本的非参数替代）

    当数据不服从正态分布时使用
    """
    baseline = np.asarray(baseline)
    variant = np.asarray(variant)

    if len(baseline) != len(variant):
        raise ValueError("Baseline and variant must have same length")
    if len(baseline) < 5:
        raise ValueError("Need at least 5 pairs for Wilcoxon test")

    statistic, p_value = stats.wilcoxon(baseline, variant, alternative=alternative)

    # 非参数效应量：秩相关系数 r = Z / sqrt(N)
    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 0
    n = len(baseline)
    r = z_score / np.sqrt(n) if n > 0 else 0

    return StatisticalTestResult(
        test_name="wilcoxon_signed_rank",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(r),
        effect_interpretation=interpret_cohens_d(r),
    )


# =============================================================================
# Mann-Whitney U检验
# =============================================================================

def mann_whitney_u_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> StatisticalTestResult:
    """
    Mann-Whitney U检验（独立样本的非参数替代）
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) < 3 or len(group2) < 3:
        raise ValueError("Need at least 3 samples per group")

    statistic, p_value = stats.mannwhitneyu(
        group1, group2,
        alternative=alternative
    )

    return StatisticalTestResult(
        test_name="mann_whitney_u",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
    )


# =============================================================================
# 方差分析 (ANOVA)
# =============================================================================

def one_way_anova(
    *groups: np.ndarray,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    单因素方差分析（One-Way ANOVA）

    用于比较三个或更多组的均值差异
    （例如：多个数据稀缺比例下的模型性能）

    Args:
        *groups: 多组样本数组
        alpha: 显著性水平

    Returns:
        StatisticalTestResult
    """
    groups = [np.asarray(g) for g in groups]

    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for ANOVA")

    for i, g in enumerate(groups):
        if len(g) < 2:
            raise ValueError(f"Group {i} has fewer than 2 samples")

    # 执行ANOVA
    statistic, p_value = stats.f_oneway(*groups)

    # 效应量：eta-squared
    all_data = np.concatenate(groups)
    ss_total = np.sum((all_data - np.mean(all_data))**2)
    group_means = [np.mean(g) for g in groups]
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (m - grand_mean)**2 for g, m in zip(groups, group_means))

    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return StatisticalTestResult(
        test_name="one_way_anova",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(eta_sq),
        effect_interpretation="small" if eta_sq < 0.06 else ("medium" if eta_sq < 0.14 else "large"),
        additional_info={
            'n_groups': len(groups),
            'group_sizes': [len(g) for g in groups],
            'group_means': [float(np.mean(g)) for g in groups],
        }
    )


def repeated_measures_anova(
    data: np.ndarray,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    重复测量方差分析

    用于比较同一组受试者在多个条件下的差异
    （例如：同一个模型在不同数据比例下的性能）

    Args:
        data: 形状为 (n_subjects, n_conditions) 的数组
        alpha: 显著性水平
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Data must be 2D array with shape (n_subjects, n_conditions)")

    n_subjects, n_conditions = data.shape

    if n_subjects < 2 or n_conditions < 2:
        raise ValueError("Need at least 2 subjects and 2 conditions")

    # 使用Friedman检验作为非参数替代（如果正态性不满足）
    # 或使用简单的区组ANOVA

    # 计算F统计量
    subject_means = np.mean(data, axis=1)
    condition_means = np.mean(data, axis=0)
    grand_mean = np.mean(data)

    ss_subjects = n_conditions * np.sum((subject_means - grand_mean)**2)
    ss_conditions = n_subjects * np.sum((condition_means - grand_mean)**2)
    ss_total = np.sum((data - grand_mean)**2)
    ss_error = ss_total - ss_subjects - ss_conditions

    df_conditions = n_conditions - 1
    df_subjects = n_subjects - 1
    df_error = (n_conditions - 1) * (n_subjects - 1)

    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error if df_error > 0 else 1e-10

    f_stat = ms_conditions / ms_error
    p_value = 1 - stats.f.cdf(f_stat, df_conditions, df_error)

    # 效应量：partial eta-squared
    partial_eta_sq = ss_conditions / (ss_conditions + ss_error)

    return StatisticalTestResult(
        test_name="repeated_measures_anova",
        statistic=float(f_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(partial_eta_sq),
        effect_interpretation="small" if partial_eta_sq < 0.06 else ("medium" if partial_eta_sq < 0.14 else "large"),
        additional_info={
            'n_subjects': n_subjects,
            'n_conditions': n_conditions,
            'condition_means': [float(m) for m in condition_means],
        }
    )


# =============================================================================
# 事后检验 (Post-hoc)
# =============================================================================

def tukey_hsd_test(
    *groups: np.ndarray,
    alpha: float = 0.05
) -> List[Dict]:
    """
    Tukey's HSD (Honestly Significant Difference) 事后检验

    ANOVA显著后，用于确定哪些组之间存在显著差异

    Returns:
        比较结果列表
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        import pandas as pd

        # 准备数据
        all_data = []
        all_groups = []
        for i, group in enumerate(groups):
            all_data.extend(group)
            all_groups.extend([f"Group_{i}"] * len(group))

        df = pd.DataFrame({'value': all_data, 'group': all_groups})
        tukey = pairwise_tukeyhsd(df['value'], df['group'], alpha=alpha)

        results = []
        for row in tukey.summary().data[1:]:
            results.append({
                'group1': row[0],
                'group2': row[1],
                'mean_diff': float(row[2]),
                'p_adj': float(row[3]),
                'significant': row[6] == True,
            })

        return results
    except ImportError:
        warnings.warn("statsmodels not available, skipping Tukey HSD")
        return []


# =============================================================================
# 综合比较函数
# =============================================================================

def compare_two_models(
    model_a_scores: List[float],
    model_b_scores: List[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05,
    paired: bool = True
) -> Dict:
    """
    综合比较两个模型（自动选择检验方法）

    Args:
        model_a_scores: 模型A的多次运行结果
        model_b_scores: 模型B的多次运行结果
        model_a_name: 模型A名称
        model_b_name: 模型B名称
        alpha: 显著性水平
        paired: 是否为配对样本

    Returns:
        比较结果字典
    """
    a = np.asarray(model_a_scores)
    b = np.asarray(model_b_scores)

    # 正态性检查
    if paired:
        diff = a - b
        normality = check_normality(diff, alpha)
    else:
        norm_a = check_normality(a, alpha)
        norm_b = check_normality(b, alpha)
        normality = {'group_a': norm_a, 'group_b': norm_b}

    # 选择检验方法
    if paired:
        if normality.get('is_normal', True):
            result = paired_t_test(a, b, alpha)
        else:
            result = wilcoxon_signed_rank_test(a, b, alpha)
    else:
        if normality.get('group_a', {}).get('is_normal', True) and \
           normality.get('group_b', {}).get('is_normal', True):
            result = independent_t_test(a, b, alpha)
        else:
            result = mann_whitney_u_test(a, b, alpha)

    # 均值差异
    mean_diff = np.mean(a) - np.mean(b)
    std_diff = np.std(a - b, ddof=1) if paired else None

    return {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'mean_a': float(np.mean(a)),
        'mean_b': float(np.mean(b)),
        'std_a': float(np.std(a, ddof=1)),
        'std_b': float(np.std(b, ddof=1)),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff) if std_diff is not None else None,
        'test_result': result.to_dict(),
        'normality_check': normality,
    }


def compare_multiple_models(
    model_results: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict:
    """
    比较多个模型（ANOVA + 事后检验）

    Args:
        model_results: {model_name: [score1, score2, ...]}
        alpha: 显著性水平

    Returns:
        比较结果字典
    """
    names = list(model_results.keys())
    groups = [np.asarray(model_results[name]) for name in names]

    # ANOVA
    anova_result = one_way_anova(*groups, alpha=alpha)

    # 事后检验（如果ANOVA显著）
    post_hoc = []
    if anova_result.significant:
        try:
            post_hoc = tukey_hsd_test(*groups, alpha=alpha)
        except Exception as e:
            warnings.warn(f"Post-hoc test failed: {e}")

    # 成对比较
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            pairwise[key] = compare_two_models(
                groups[i], groups[j],
                names[i], names[j],
                alpha, paired=False
            )

    return {
        'anova': anova_result.to_dict(),
        'post_hoc': post_hoc,
        'pairwise': pairwise,
        'model_means': {name: float(np.mean(g)) for name, g in zip(names, groups)},
        'model_stds': {name: float(np.std(g, ddof=1)) for name, g in zip(names, groups)},
    }


def compare_data_scarcity_curves(
    model_curves: Dict[str, Dict[float, List[float]]],
    alpha: float = 0.05
) -> Dict:
    """
    比较不同模型的数据稀缺性曲线

    Args:
        model_curves: {
            'model_name': {
                ratio_1: [seed1_acc, seed2_acc, ...],
                ratio_2: [seed1_acc, seed2_acc, ...],
                ...
            }
        }
        alpha: 显著性水平

    Returns:
        比较结果
    """
    results = {}

    # 对每个比例进行比较
    ratios = sorted(next(iter(model_curves.values())).keys())

    for ratio in ratios:
        ratio_data = {}
        for model_name, curves in model_curves.items():
            ratio_data[model_name] = curves[ratio]

        results[f"ratio_{ratio:.2f}"] = compare_multiple_models(ratio_data, alpha)

    # 识别临界点：Hybrid开始显著优于CNN的比例
    critical_points = {}
    for ratio in ratios:
        key = f"ratio_{ratio:.2f}"
        pairwise = results[key].get('pairwise', {})
        for comp_key, comp_result in pairwise.items():
            if 'hybrid' in comp_key.lower() and 'resnet' in comp_key.lower():
                if comp_result['test_result']['significant']:
                    if 'hybrid' not in critical_points:
                        critical_points['hybrid'] = ratio
                    break

    results['critical_points'] = critical_points
    return results


# =============================================================================
# 报告生成
# =============================================================================

def generate_statistical_report(
    test_results: List[StatisticalTestResult],
    output_path: Optional[str] = None
) -> str:
    """
    生成统计检验报告

    Args:
        test_results: 检验结果列表
        output_path: 输出文件路径（可选）

    Returns:
        报告字符串
    """
    lines = [
        "# 统计显著性检验报告",
        "",
        f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 检验结果汇总",
        "",
        "| 检验名称 | 统计量 | p值 | 显著性 | 效应量 | 解释 |",
        "|---------|--------|-----|--------|--------|------|",
    ]

    for result in test_results:
        sig = "**显著**" if result.significant else "不显著"
        es = f"{result.effect_size:.3f}" if result.effect_size else "N/A"
        interp = result.effect_interpretation or "N/A"
        lines.append(
            f"| {result.test_name} | {result.statistic:.4f} | {result.p_value:.6f} | {sig} | {es} | {interp} |"
        )

    lines.extend([
        "",
        "## 显著性标记",
        "",
        "- *** p < 0.001 (高度显著)",
        "- **  p < 0.01 (非常显著)",
        "- *   p < 0.05 (显著)",
        "- ns  p >= 0.05 (不显著)",
        "",
        "## 效应量解释（Cohen's d）",
        "",
        "| 范围 | 解释 |",
        "|------|------|",
        "| |d| < 0.2 | 可忽略 |",
        "| 0.2 ≤ |d| < 0.5 | 小 |",
        "| 0.5 ≤ |d| < 0.8 | 中等 |",
        "| |d| ≥ 0.8 | 大 |",
    ])

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

    return report


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("Testing statistical_tests module...")

    # 测试数据：模拟两个模型的多次运行结果
    np.random.seed(42)
    resnet_scores = np.random.normal(0.85, 0.02, 10)
    hybrid_scores = np.random.normal(0.88, 0.015, 10)

    # 1. 配对t检验
    print("\n1. Paired t-test:")
    result = paired_t_test(resnet_scores, hybrid_scores)
    print(f"   {result}")

    # 2. 独立样本t检验
    print("\n2. Welch's t-test:")
    result = independent_t_test(resnet_scores, hybrid_scores)
    print(f"   {result}")

    # 3. Wilcoxon检验
    print("\n3. Wilcoxon signed-rank:")
    result = wilcoxon_signed_rank_test(resnet_scores, hybrid_scores)
    print(f"   {result}")

    # 4. 多组ANOVA
    print("\n4. One-way ANOVA:")
    vit_scores = np.random.normal(0.86, 0.018, 10)
    result = one_way_anova(resnet_scores, hybrid_scores, vit_scores)
    print(f"   {result}")

    # 5. 综合比较
    print("\n5. Comprehensive comparison (2 models):")
    comp = compare_two_models(
        resnet_scores, hybrid_scores,
        "ResNet50", "Hybrid-Advanced",
        paired=True
    )
    print(f"   Mean diff: {comp['mean_diff']:.4f}")
    print(f"   p-value: {comp['test_result']['p_value']:.6f}")

    # 6. 多模型比较
    print("\n6. Multiple models comparison:")
    multi = compare_multiple_models({
        'ResNet50': resnet_scores,
        'ViT': vit_scores,
        'Hybrid': hybrid_scores,
    })
    print(f"   ANOVA p: {multi['anova']['p_value']:.6f}")
    for model, mean in multi['model_means'].items():
        print(f"   {model}: {mean:.4f} ± {multi['model_stds'][model]:.4f}")

    # 7. 生成报告
    print("\n7. Generate report:")
    report = generate_statistical_report([
        paired_t_test(resnet_scores, hybrid_scores),
        one_way_anova(resnet_scores, hybrid_scores, vit_scores),
    ])
    print(report[:500] + "...")

    print("\nAll statistical test module tests passed!")
