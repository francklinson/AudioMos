"""
降噪算法统计显著性检验模块

提供算法之间性能差异的统计显著性分析能力。
支持:
- 配对t检验 (Paired t-test)
- Wilcoxon符号秩检验 (非参数)
- Bootstrap置信区间
- Cohen's d效应量
- 多重比较校正 (Bonferroni / Holm-Bonferroni)
- 综合比较报告
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ===========================
# 数据类
# ===========================


@dataclass
class TTestResult:
    """配对t检验结果"""
    statistic: float
    p_value: float
    mean_diff: float
    ci_95: Tuple[float, float]
    significant: bool  # p < 0.05
    degrees_of_freedom: int


@dataclass
class WilcoxonResult:
    """Wilcoxon符号秩检验结果"""
    statistic: float
    p_value: float
    significant: bool


@dataclass
class EffectSize:
    """效应量"""
    cohens_d: float
    interpretation: str  # 'negligible' / 'small' / 'medium' / 'large'
    hedges_g: Optional[float] = None  # 小样本校正版


@dataclass
class MetricComparison:
    """单个指标的算法对比"""
    metric_name: str
    scores_a: List[float]
    scores_b: List[float]
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    t_test: Optional[TTestResult] = None
    wilcoxon: Optional[WilcoxonResult] = None
    effect_size: Optional[EffectSize] = None
    bootstrap_ci_a: Optional[Tuple[float, float]] = None
    bootstrap_ci_b: Optional[Tuple[float, float]] = None


@dataclass
class ComparisonReport:
    """完整对比报告"""
    algorithm_a: str
    algorithm_b: str
    n_samples: int
    metrics: Dict[str, MetricComparison] = field(default_factory=dict)
    multiple_comparison_corrected: bool = False
    correction_method: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "algorithm_a": self.algorithm_a,
            "algorithm_b": self.algorithm_b,
            "n_samples": self.n_samples,
            "metrics": {
                name: {
                    "mean_a": mc.mean_a,
                    "mean_b": mc.mean_b,
                    "std_a": mc.std_a,
                    "std_b": mc.std_b,
                    "mean_diff": mc.t_test.mean_diff if mc.t_test else None,
                    "p_value": mc.t_test.p_value if mc.t_test else None,
                    "significant": mc.t_test.significant if mc.t_test else None,
                    "cohens_d": mc.effect_size.cohens_d if mc.effect_size else None,
                    "ci_95_a": list(mc.bootstrap_ci_a) if mc.bootstrap_ci_a else None,
                    "ci_95_b": list(mc.bootstrap_ci_b) if mc.bootstrap_ci_b else None,
                }
                for name, mc in self.metrics.items()
            },
            "summary": self.summary,
        }


@dataclass
class FullComparisonMatrix:
    """多算法全比较矩阵"""
    algorithms: List[str]
    metric_comparisons: Dict[str, Dict[Tuple[str, str], MetricComparison]] = field(
        default_factory=dict
    )
    # metric_comparisons[metric_name][(algo_a, algo_b)] = MetricComparison


# ===========================
# 统计分析器
# ===========================


class StatisticalAnalyzer:
    """
    降噪算法统计显著性分析器

    使用方式:
        analyzer = StatisticalAnalyzer()
        report = analyzer.compare_algorithms(
            algo_a_scores={"pesq": [2.1, 2.3, ...], "stoi": [0.85, 0.87, ...]},
            algo_b_scores={"pesq": [2.5, 2.6, ...], "stoi": [0.88, 0.89, ...]},
            name_a="MetricGAN+", name_b="FRCRN"
        )
        print(report.summary)
    """

    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """
        初始化分析器

        Args:
            alpha: 显著性水平（默认0.05）
            random_seed: 随机种子（用于bootstrap）
        """
        self.alpha = alpha
        self.random_seed = random_seed
        np.random.seed(random_seed)

    # ===========================
    # 核心检验方法
    # ===========================

    def paired_t_test(self, scores_a: np.ndarray, scores_b: np.ndarray) -> TTestResult:
        """
        配对t检验

        检验两个算法在同一组测试文件上的性能差异是否统计显著。

        Args:
            scores_a: 算法A的得分数组
            scores_b: 算法B的得分数组

        Returns:
            TTestResult
        """
        from scipy import stats

        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)

        # 移除缺失值对
        valid_mask = ~(np.isnan(scores_a) | np.isnan(scores_b) |
                       np.isinf(scores_a) | np.isinf(scores_b))
        scores_a = scores_a[valid_mask]
        scores_b = scores_b[valid_mask]

        if len(scores_a) < 3:
            return TTestResult(
                statistic=np.nan, p_value=1.0, mean_diff=np.nan,
                ci_95=(np.nan, np.nan), significant=False,
                degrees_of_freedom=len(scores_a) - 1,
            )

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        mean_diff = np.mean(scores_b) - np.mean(scores_a)
        diffs = scores_b - scores_a
        ci = stats.t.interval(
            0.95, len(diffs) - 1,
            loc=np.mean(diffs), scale=stats.sem(diffs)
        )

        return TTestResult(
            statistic=float(t_stat),
            p_value=float(p_value),
            mean_diff=float(mean_diff),
            ci_95=(float(ci[0]), float(ci[1])),
            significant=bool(p_value < self.alpha),
            degrees_of_freedom=len(scores_a) - 1,
        )

    def wilcoxon_test(self, scores_a: np.ndarray, scores_b: np.ndarray) -> WilcoxonResult:
        """
        Wilcoxon符号秩检验（非参数检验）

        当指标分布不满足正态性时使用。

        Args:
            scores_a: 算法A的得分数组
            scores_b: 算法B的得分数组

        Returns:
            WilcoxonResult
        """
        from scipy import stats

        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)

        # 移除缺失值对
        valid_mask = ~(np.isnan(scores_a) | np.isnan(scores_b) |
                       np.isinf(scores_a) | np.isinf(scores_b))
        scores_a = scores_a[valid_mask]
        scores_b = scores_b[valid_mask]

        if len(scores_a) < 5:
            return WilcoxonResult(statistic=np.nan, p_value=1.0, significant=False)

        try:
            w_stat, p_value = stats.wilcoxon(scores_a, scores_b)
        except Exception:
            # 如果所有差值为0，回退
            return WilcoxonResult(statistic=0.0, p_value=1.0, significant=False)

        return WilcoxonResult(
            statistic=float(w_stat),
            p_value=float(p_value),
            significant=bool(p_value < self.alpha),
        )

    def bootstrap_confidence_interval(
        self, scores: np.ndarray, n_iterations: int = 10000, ci: float = 0.95
    ) -> Tuple[float, float]:
        """
        Bootstrap置信区间

        Args:
            scores: 得分数组
            n_iterations: bootstrap迭代次数
            ci: 置信水平

        Returns:
            (下界, 上界)
        """
        scores = np.asarray(scores, dtype=np.float64)
        scores = scores[~(np.isnan(scores) | np.isinf(scores))]

        if len(scores) < 3:
            return float(np.mean(scores)) if len(scores) > 0 else np.nan, np.nan

        boot_means = []
        n = len(scores)

        for _ in range(n_iterations):
            sample_idx = np.random.randint(0, n, n)
            boot_sample = scores[sample_idx]
            boot_means.append(np.mean(boot_sample))

        alpha = (1 - ci) / 2
        lower = np.percentile(boot_means, alpha * 100)
        upper = np.percentile(boot_means, (1 - alpha) * 100)

        return float(lower), float(upper)

    def cohens_d(self, scores_a: np.ndarray, scores_b: np.ndarray, paired: bool = True) -> EffectSize:
        """
        计算 Cohen's d 效应量

        Args:
            scores_a: 算法A的得分
            scores_b: 算法B的得分
            paired: 是否是配对设计

        Returns:
            EffectSize
        """
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)

        valid_mask = ~(np.isnan(scores_a) | np.isnan(scores_b) |
                       np.isinf(scores_a) | np.isinf(scores_b))
        scores_a = scores_a[valid_mask]
        scores_b = scores_b[valid_mask]

        if len(scores_a) < 3:
            return EffectSize(cohens_d=0.0, interpretation="negligible")

        mean_diff = np.mean(scores_b) - np.mean(scores_a)

        if paired:
            # 配对设计: 使用差值的标准差
            diffs = scores_b - scores_a
            pooled_std = np.std(diffs, ddof=1)
        else:
            # 独立设计: 使用合并标准差
            n_a, n_b = len(scores_a), len(scores_b)
            pooled_var = ((n_a - 1) * np.var(scores_a, ddof=1) + (n_b - 1) * np.var(scores_b, ddof=1)) / (n_a + n_b - 2)
            pooled_std = np.sqrt(pooled_var)

        if pooled_std < 1e-10:
            d = 0.0
        else:
            d = mean_diff / pooled_std

        # Hedges' g 小样本校正
        n = len(scores_a)
        hedges_g = d * (1 - 3 / (4 * n - 9)) if n > 3 else d

        # 效应量解释
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return EffectSize(
            cohens_d=float(d),
            interpretation=interpretation,
            hedges_g=float(hedges_g),
        )

    # ===========================
    # 多重比较校正
    # ===========================

    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Bonferroni多重比较校正"""
        n = len(p_values)
        return [min(1.0, p * n) for p in p_values]

    def holm_bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Holm-Bonferroni校正（比Bonferroni更不保守）"""
        n = len(p_values)
        # 排序p值（保持原始索引）
        indexed = list(enumerate(p_values))
        indexed.sort(key=lambda x: x[1])
        adjusted = [0.0] * n

        for k, (orig_idx, p_val) in enumerate(indexed):
            adjusted[orig_idx] = min(1.0, p_val * (n - k))

        return adjusted

    def multiple_comparison_correction(
        self, p_values: List[float], method: str = "bonferroni"
    ) -> List[float]:
        """
        多重比较校正

        Args:
            p_values: 原始p值列表
            method: 校正方法 ('bonferroni' / 'holm' / 'none')

        Returns:
            校正后的p值列表
        """
        if method == "holm":
            return self.holm_bonferroni_correction(p_values)
        elif method == "bonferroni":
            return self.bonferroni_correction(p_values)
        else:
            return p_values

    # ===========================
    # 综合方法
    # ===========================

    def compare_algorithms(
        self,
        algo_a_scores: Dict[str, List[float]],
        algo_b_scores: Dict[str, List[float]],
        name_a: str = "Algorithm A",
        name_b: str = "Algorithm B",
        metrics_to_compare: Optional[List[str]] = None,
    ) -> ComparisonReport:
        """
        两个算法的完整对比分析

        Args:
            algo_a_scores: {metric_name: [scores]} 算法A的各指标得分
            algo_b_scores: {metric_name: [scores]} 算法B的各指标得分
            name_a: 算法A名称
            name_b: 算法B名称
            metrics_to_compare: 要比较的指标列表（None=所有共同指标）

        Returns:
            ComparisonReport
        """
        if metrics_to_compare is None:
            metrics_to_compare = sorted(
                set(algo_a_scores.keys()) & set(algo_b_scores.keys())
            )

        report = ComparisonReport(
            algorithm_a=name_a,
            algorithm_b=name_b,
            n_samples=len(next(iter(algo_a_scores.values()), [])),
        )

        for metric in metrics_to_compare:
            scores_a = algo_a_scores.get(metric, [])
            scores_b = algo_b_scores.get(metric, [])

            if len(scores_a) < 3 or len(scores_b) < 3:
                continue

            sa = np.array(scores_a, dtype=np.float64)
            sb = np.array(scores_b, dtype=np.float64)

            mc = MetricComparison(
                metric_name=metric,
                scores_a=list(sa),
                scores_b=list(sb),
                mean_a=float(np.mean(sa[~(np.isnan(sa) | np.isinf(sa))])),
                mean_b=float(np.mean(sb[~(np.isnan(sb) | np.isinf(sb))])),
                std_a=float(np.std(sa[~(np.isnan(sa) | np.isinf(sa))])),
                std_b=float(np.std(sb[~(np.isnan(sb) | np.isinf(sb))])),
            )

            # t检验
            mc.t_test = self.paired_t_test(sa, sb)
            # Wilcoxon
            mc.wilcoxon = self.wilcoxon_test(sa, sb)
            # 效应量
            mc.effect_size = self.cohens_d(sa, sb, paired=True)
            # Bootstrap CI
            mc.bootstrap_ci_a = self.bootstrap_confidence_interval(sa)
            mc.bootstrap_ci_b = self.bootstrap_confidence_interval(sb)

            report.metrics[metric] = mc

        # 生成摘要
        report.summary = self._generate_summary(report)

        return report

    def full_pairwise_comparison(
        self,
        algorithm_scores: Dict[str, Dict[str, List[float]]],
        metrics: Optional[List[str]] = None,
        correction_method: str = "bonferroni",
    ) -> FullComparisonMatrix:
        """
        所有算法的两两全比较

        Args:
            algorithm_scores: {algo_name: {metric: [scores]}}
            metrics: 指标列表
            correction_method: 多重比较校正方法

        Returns:
            FullComparisonMatrix
        """
        if metrics is None:
            # 收集所有共同指标
            all_metrics = set()
            for scores in algorithm_scores.values():
                all_metrics.update(scores.keys())
            metrics = sorted(all_metrics)

        algos = sorted(algorithm_scores.keys())
        matrix = FullComparisonMatrix(algorithms=algos)

        for metric in metrics:
            matrix.metric_comparisons[metric] = {}

            # 收集所有p值用于校正
            p_values = []
            comparison_keys = []

            for i, algo_a in enumerate(algos):
                for j, algo_b in enumerate(algos):
                    if i >= j:
                        continue

                    scores_a = algorithm_scores.get(algo_a, {}).get(metric, [])
                    scores_b = algorithm_scores.get(algo_b, {}).get(metric, [])

                    if len(scores_a) < 3 or len(scores_b) < 3:
                        continue

                    sa = np.array(scores_a, dtype=np.float64)
                    sb = np.array(scores_b, dtype=np.float64)

                    mc = MetricComparison(
                        metric_name=metric,
                        scores_a=list(sa),
                        scores_b=list(sb),
                        mean_a=float(np.mean(sa[~(np.isnan(sa) | np.isinf(sa))])),
                        mean_b=float(np.mean(sb[~(np.isnan(sb) | np.isinf(sb))])),
                        std_a=float(np.std(sa[~(np.isnan(sa) | np.isinf(sa))])),
                        std_b=float(np.std(sb[~(np.isnan(sb) | np.isinf(sb))])),
                    )
                    mc.t_test = self.paired_t_test(sa, sb)
                    mc.effect_size = self.cohens_d(sa, sb)

                    comparison_keys.append((algo_a, algo_b))
                    matrix.metric_comparisons[metric][(algo_a, algo_b)] = mc
                    p_values.append(mc.t_test.p_value)

            # 多重比较校正
            corrected_p = self.multiple_comparison_correction(p_values, correction_method)

            for (algo_a, algo_b), new_p in zip(comparison_keys, corrected_p):
                mc = matrix.metric_comparisons[metric][(algo_a, algo_b)]
                if mc.t_test:
                    mc.t_test.p_value = new_p
                    mc.t_test.significant = new_p < self.alpha
            matrix.metric_comparisons[metric] = {
                k: v for k, v in matrix.metric_comparisons[metric].items()
            }

            matrix.multiple_comparison_corrected = True
            matrix.correction_method = correction_method

        return matrix

    def check_normality(self, scores: np.ndarray, method: str = "shapiro") -> Dict:
        """
        检验数据是否符合正态分布

        Args:
            scores: 得分数组
            method: 检验方法 ('shapiro' / 'dagostino')

        Returns:
            {statistic, p_value, is_normal}
        """
        from scipy import stats

        scores = np.asarray(scores, dtype=np.float64)
        scores = scores[~(np.isnan(scores) | np.isinf(scores))]

        if len(scores) < 3:
            return {"statistic": np.nan, "p_value": 1.0, "is_normal": False}

        if method == "dagostino" and len(scores) >= 20:
            stat, p = stats.normaltest(scores)
        else:
            stat, p = stats.shapiro(scores)

        return {
            "statistic": float(stat),
            "p_value": float(p),
            "is_normal": bool(p > 0.05),
        }

    # ===========================
    # 辅助方法
    # ===========================

    def _generate_summary(self, report: ComparisonReport) -> str:
        """生成人类可读的对比摘要"""
        lines = []
        lines.append(f"算法对比: {report.algorithm_a} vs {report.algorithm_b}")
        lines.append(f"样本数: {report.n_samples}")
        lines.append("")

        significant_metrics = []
        for metric, mc in report.metrics.items():
            if mc.t_test and mc.t_test.significant:
                direction = "↑" if mc.mean_b > mc.mean_a else "↓"
                significant_metrics.append(
                    f"  {metric}: {mc.mean_a:.3f} → {mc.mean_b:.3f} {direction} "
                    f"(p={mc.t_test.p_value:.4f}, d={mc.effect_size.cohens_d:.2f})"
                )

        if significant_metrics:
            lines.append(f"显著性指标 (p<{self.alpha}):")
            lines.extend(significant_metrics)
        else:
            lines.append(f"无显著差异 (p≥{self.alpha})")

        return "\n".join(lines)
