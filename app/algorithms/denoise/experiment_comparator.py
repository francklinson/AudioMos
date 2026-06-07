"""
实验对比与历史追踪模块

跨实验的算法性能追踪和对比分析。
- 同一算法的历史趋势
- 不同实验配置下的性能对比
- 最优配置查找
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from .experiment_db import ExperimentDB, ExperimentSummary

logger = logging.getLogger(__name__)


@dataclass
class TrendPoint:
    """趋势数据点"""
    experiment_id: str
    experiment_name: str
    created_at: str
    metric_name: str
    mean_value: float
    std_value: float


@dataclass
class TrendReport:
    """趋势分析报告"""
    algorithm_name: str
    metric: str
    data_points: List[TrendPoint] = field(default_factory=list)

    @property
    def improvement_rate(self) -> Optional[float]:
        """改进趋势（简单线性回归斜率）"""
        if len(self.data_points) < 2:
            return None
        xs = range(len(self.data_points))
        ys = [p.mean_value for p in self.data_points]
        return float(np.polyfit(xs, ys, 1)[0])  # 斜率

    @property
    def best_score(self) -> float:
        if not self.data_points:
            return 0.0
        return max(p.mean_value for p in self.data_points)

    @property
    def latest_score(self) -> float:
        if not self.data_points:
            return 0.0
        return self.data_points[-1].mean_value


@dataclass
class ComparisonResult:
    """实验对比结果"""
    experiment_ids: List[str]
    metric_comparisons: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # metric_comparisons[metric][experiment_id][algo_name] = score


class ExperimentComparator:
    """
    实验对比器

    使用方式:
        comparator = ExperimentComparator()
        trends = comparator.get_algorithm_trend("speechbrain_metricgan", "pesq")
        comparison = comparator.compare_experiments(["exp_1", "exp_2"])
    """

    def __init__(self, db_path: str = "./data/experiments/results.db"):
        self.db = ExperimentDB(db_path)

    def get_algorithm_trend(
        self, algorithm_name: str, metric: str = "pesq"
    ) -> TrendReport:
        """
        获取算法在多次实验中的性能趋势

        Args:
            algorithm_name: 算法名称
            metric: 关注的指标

        Returns:
            TrendReport
        """
        history = self.db.get_algorithm_history(algorithm_name, metric)

        report = TrendReport(algorithm_name=algorithm_name, metric=metric)

        for entry in history:
            report.data_points.append(TrendPoint(
                experiment_id=entry["experiment_id"],
                experiment_name=entry["name"],
                created_at=entry["created_at"],
                metric_name=entry["metric_name"],
                mean_value=entry["mean_value"],
                std_value=entry.get("std_value", 0),
            ))

        return report

    def compare_experiments(
        self, experiment_ids: List[str], metrics: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        对比多个实验的各算法性能

        Args:
            experiment_ids: 实验ID列表
            metrics: 关注的指标列表

        Returns:
            ComparisonResult
        """
        if metrics is None:
            metrics = ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]

        db_comparison = self.db.compare_experiments(experiment_ids, metrics)

        result = ComparisonResult(experiment_ids=experiment_ids)
        result.metric_comparisons = db_comparison["metrics_comparison"]

        return result

    def find_best_configuration(
        self, algorithm: str, target_metric: str = "pesq"
    ) -> Optional[Dict]:
        """
        查找某算法在历史实验中达到最优性能的配置

        Args:
            algorithm: 算法名称
            target_metric: 目标指标

        Returns:
            最优实验配置信息
        """
        history = self.db.get_algorithm_history(algorithm, target_metric)

        if not history:
            return None

        best = max(history, key=lambda x: x.get("mean_value", 0) or 0)

        # 获取完整实验信息
        exp = self.db.get_experiment(best["experiment_id"])

        if exp:
            return {
                "experiment_id": best["experiment_id"],
                "experiment_name": best["name"],
                "metric": target_metric,
                "best_score": best["mean_value"],
                "created_at": best["created_at"],
                "config": exp.get("config_json", {}),
                "dataset": best.get("dataset", ""),
            }

        return None

    def rank_algorithms_across_experiments(
        self, metric: str = "pesq", limit: int = 10
    ) -> List[Dict]:
        """
        在所有实验中综合排名算法

        Args:
            metric: 排名指标
            limit: 返回数量

        Returns:
            排名列表
        """
        return self.db.find_best_overall(metric, limit)

    def generate_improvement_report(self) -> str:
        """
        生成改进总结报告（Markdown格式）

        Returns:
            Markdown字符串
        """
        lines = ["# 降噪算法改进追踪报告\n"]
        lines.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 所有跟踪的算法
        best_algos = self.db.find_best_overall("pesq", limit=10)

        lines.append("## 历史最佳算法 (PESQ)\n")
        lines.append("| 排名 | 算法 | 最佳PESQ | 实验次数 |")
        lines.append("|------|------|---------|----------|")

        for i, entry in enumerate(best_algos, 1):
            lines.append(f"| {i} | {entry['algorithm_name']} | {entry['best_score']:.3f} | {entry['experiment_count']} |")

        lines.append("")

        # 各算法趋势
        for entry in best_algos[:5]:
            algo = entry["algorithm_name"]
            trend = self.get_algorithm_trend(algo, "pesq")
            if len(trend.data_points) >= 2:
                direction = "📈 提升" if trend.improvement_rate and trend.improvement_rate > 0 else "📉 下降"
                lines.append(f"- **{algo}**: {direction} "
                           f"(趋势斜率: {trend.improvement_rate:.4f}/实验, "
                           f"最新: {trend.latest_score:.3f}, 最佳: {trend.best_score:.3f})")

        return "\n".join(lines)
