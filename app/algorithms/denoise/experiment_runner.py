"""
降噪测评实验运行器

统一管理测评实验的完整生命周期:
- 实验配置 (ExperimentConfig)
- 实验执行 (ExperimentRunner)
- 一键复现 (reproduce_experiment)
"""

import os
import sys
import time
import json
import logging
import subprocess
import uuid
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .dataset_manager import DatasetManager
from .evaluator import DenoiseEvaluator
from .benchmark import BenchmarkRunner, BenchmarkResult
from .significance import StatisticalAnalyzer, ComparisonReport
from .visualizer import BenchmarkVisualizer
from .efficiency_profiler import EfficiencyProfiler, EfficiencyReport
from .experiment_db import ExperimentDB
from .registry import DenoiserRegistry

logger = logging.getLogger(__name__)


# ===========================
# 数据结构
# ===========================


@dataclass
class EvaluatorConfig:
    """评估器配置"""
    metrics: List[str] = field(default_factory=lambda: [
        "pesq", "stoi", "sisdr", "dnsmos_ovrl", "nisqa_mos", "utmos"
    ])
    sample_rate: int = 16000
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str = ""
    algorithms: List[str] = field(default_factory=list)  # 算法名称列表
    dataset_key: str = "builtin"  # 数据集键名
    dataset_config: Optional[Dict] = None  # 数据集配置覆盖
    evaluator_config: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    significance_test: bool = True
    visualization: bool = True
    efficiency_profile: bool = False
    random_seed: int = 42
    output_dir: str = "./data/experiments"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "algorithms": self.algorithms,
            "dataset_key": self.dataset_key,
            "dataset_config": self.dataset_config,
            "significance_test": self.significance_test,
            "visualization": self.visualization,
            "efficiency_profile": self.efficiency_profile,
            "random_seed": self.random_seed,
            "tags": self.tags,
        }


@dataclass
class ExperimentResult:
    """实验完整结果"""
    experiment_id: str
    config: ExperimentConfig
    benchmark_result: Optional[Any] = None  # BenchmarkResult
    statistical_report: Optional[ComparisonReport] = None
    efficiency_report: Optional[Dict[str, EfficiencyReport]] = None
    chart_files: List[str] = field(default_factory=list)
    created_at: str = ""
    duration: float = 0.0
    git_commit_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "duration": self.duration,
            "git_commit_hash": self.git_commit_hash,
            "chart_files": self.chart_files,
        }


# ===========================
# 实验运行器
# ===========================


class ExperimentRunner:
    """
    实验运行器

    统一编排测评流程: 数据集准备 → 多算法降噪 → 评估 → 统计 → 可视化 → 持久化

    使用方式:
        runner = ExperimentRunner()
        config = ExperimentConfig(
            name="DNS基准测试",
            algorithms=["speechbrain_metricgan", "clearervoice_frcrn"],
            dataset_key="dns_challenge",
            significance_test=True,
        )
        result = runner.run(config)
    """

    def __init__(
        self,
        db_path: str = "./data/experiments/results.db",
        storage_dir: str = "./data/experiments",
    ):
        """
        初始化实验运行器

        Args:
            db_path: 实验结果数据库路径
            storage_dir: 实验文件存储目录
        """
        self.db = ExperimentDB(db_path)
        self.storage_dir = storage_dir
        self.dataset_manager = DatasetManager(
            storage_dir=os.path.join(os.path.dirname(storage_dir), "datasets")
        )
        self.visualizer = BenchmarkVisualizer(
            output_dir=os.path.join(storage_dir, "charts")
        )
        self.analyzer = StatisticalAnalyzer()
        self.profiler = EfficiencyProfiler()

    def run(self, config: ExperimentConfig, progress_callback: Optional[Callable] = None) -> ExperimentResult:
        """
        执行完整实验

        Args:
            config: 实验配置
            progress_callback: 进度回调 fn(stage: str, progress: int, total: int)

        Returns:
            ExperimentResult
        """
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"实验开始: {config.name} ({experiment_id})")
        logger.info(f"算法: {config.algorithms}")
        logger.info(f"数据集: {config.dataset_key}")
        logger.info("=" * 60)

        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            git_commit_hash=self._get_git_hash(),
        )

        exp_dir = os.path.join(self.storage_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)

        try:
            # Stage 1: 准备数据
            self._update_progress(progress_callback, "准备数据", 1, 5)
            sample_pairs = self._prepare_dataset(config)

            if not sample_pairs:
                raise RuntimeError("无法准备测试数据")

            noisy_files = [p.noisy_path for p in sample_pairs]
            ref_files = [p.clean_path for p in sample_pairs if p.clean_path]

            logger.info(f"数据准备完成: {len(noisy_files)} 个测试文件, "
                       f"{len(ref_files)} 个参考文件")

            # Stage 2: 运行基准测试
            self._update_progress(progress_callback, "基准测试", 2, 5)
            benchmark_runner = BenchmarkRunner(
                output_dir=os.path.join(exp_dir, "results"),
                device=config.evaluator_config.device,
            )

            if config.algorithms is None or len(config.algorithms) == 0:
                config.algorithms = DenoiserRegistry.list_denoisers()

            benchmark_result = benchmark_runner.run_benchmark(
                config.algorithms, noisy_files, ref_files if ref_files else None,
            )
            result.benchmark_result = benchmark_result

            # Stage 3: 统计检验
            if config.significance_test and len(config.algorithms) > 1:
                self._update_progress(progress_callback, "统计检验", 3, 5)
                algo_scores = self._extract_algorithm_scores(benchmark_result)
                if algo_scores and len(algo_scores) >= 2:
                    algo_names = list(algo_scores.keys())
                    report = self.analyzer.compare_algorithms(
                        algo_scores[algo_names[0]],
                        algo_scores[algo_names[1]],
                        name_a=algo_names[0],
                        name_b=algo_names[1],
                    )
                    result.statistical_report = report

            # Stage 4: 可视化
            if config.visualization:
                self._update_progress(progress_callback, "生成可视化", 4, 5)
                chart_dir = os.path.join(exp_dir, "charts")
                os.makedirs(chart_dir, exist_ok=True)

                chart_files = self._generate_charts(benchmark_result, chart_dir)
                result.chart_files = chart_files

                # 生成仪表盘
                algo_summary = {}
                for algo, algo_result in benchmark_result.algorithms.items():
                    algo_summary[algo] = algo_result.metrics

                dashboard_path = self.visualizer.generate_dashboard_html(
                    algo_summary,
                    chart_files=chart_files,
                    title=f"{config.name} - 测评仪表盘",
                    save_path=os.path.join(exp_dir, "dashboard.html"),
                )
                result.chart_files.append(dashboard_path)

            # Stage 5: 效率分析 (可选)
            if config.efficiency_profile:
                self._update_progress(progress_callback, "效率分析", 5, 5)
                efficiency_results = {}
                for algo_name in config.algorithms:
                    denoiser = DenoiserRegistry.get(algo_name)
                    if denoiser and denoiser.is_initialized():
                        report = self.profiler.profile_batch(denoiser, noisy_files[:5])
                        efficiency_results[algo_name] = report
                result.efficiency_report = efficiency_results

            # 持久化
            self._save_experiment(result)
            self._save_benchmark_scores(experiment_id, benchmark_result)

            # 保存配置快照
            config_path = os.path.join(exp_dir, "config.yaml")
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
                f.write("\n")

            result.duration = time.time() - start_time

            logger.info("=" * 60)
            logger.info(f"实验完成: {config.name} ({experiment_id})")
            logger.info(f"耗时: {result.duration:.1f}s")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"实验失败: {e}", exc_info=True)
            result.duration = time.time() - start_time
            # 保存失败状态
            self._save_experiment(result, status="failed")

        return result

    def _prepare_dataset(self, config: ExperimentConfig):
        """准备数据集"""
        from .datasets.base import EvaluationSetConfig

        ds_config = EvaluationSetConfig()
        if config.dataset_config:
            for key, value in config.dataset_config.items():
                if hasattr(ds_config, key):
                    setattr(ds_config, key, value)

        ds_config.n_samples = config.dataset_config.get("n_samples", 30) if config.dataset_config else 30
        ds_config.seed = config.random_seed

        # 确保数据集已下载
        if config.dataset_key != "builtin":
            self.dataset_manager.download_if_needed(config.dataset_key)

        return self.dataset_manager.prepare_evaluation_set(config.dataset_key, ds_config)

    def _extract_algorithm_scores(self, benchmark_result) -> Dict[str, Dict[str, List[float]]]:
        """从benchmark结果提取算法得分"""
        algo_scores = {}

        for algo_name, algo_result in benchmark_result.algorithms.items():
            scores = {}
            for metric, stats in algo_result.metrics.items():
                # 从details中重建原始得分（用于配对检验）
                metric_values = []
                for detail in algo_result.details:
                    if hasattr(detail, 'metrics'):
                        val = getattr(detail.metrics, metric, None)
                    else:
                        val = getattr(detail, metric, None)
                    if val is not None:
                        metric_values.append(val)

                if metric_values:
                    scores[metric] = metric_values

            if scores:
                algo_scores[algo_name] = scores

        return algo_scores

    def _generate_charts(self, benchmark_result, chart_dir: str) -> List[str]:
        """生成可视化图表"""
        chart_files = []

        # 提取得分
        algo_scores = {}
        for algo, algo_result in benchmark_result.algorithms.items():
            algo_scores[algo] = {}
            for detail in algo_result.details:
                if hasattr(detail, 'metrics'):
                    m = detail.metrics
                else:
                    m = detail

                for metric in ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]:
                    val = getattr(m, metric, None)
                    if val is not None:
                        algo_scores[algo].setdefault(metric, []).append(val)

        # 各指标箱线图
        for metric in ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]:
            scores_dict = {}
            for algo, scores in algo_scores.items():
                if metric in scores:
                    scores_dict[algo] = scores[metric]

            if len(scores_dict) > 1:
                path = self.visualizer.box_plot(
                    scores_dict, metric,
                    save_path=os.path.join(chart_dir, f"box_{metric}.png"),
                )
                if path:
                    chart_files.append(path)

        # 雷达图
        algo_means = {}
        for algo, algo_result in benchmark_result.algorithms.items():
            algo_means[algo] = {}
            for metric, stats in algo_result.metrics.items():
                algo_means[algo][metric] = stats.get("mean", 0)

        if len(algo_means) > 1:
            path = self.visualizer.radar_chart(
                algo_means,
                save_path=os.path.join(chart_dir, "radar.png"),
            )
            if path:
                chart_files.append(path)

        # 柱状图
        for metric in ["pesq", "stoi", "sisdr"]:
            summary_dict = {}
            for algo, algo_result in benchmark_result.algorithms.items():
                if metric in algo_result.metrics:
                    summary_dict[algo] = {metric: algo_result.metrics[metric]}

            if len(summary_dict) > 1:
                path = self.visualizer.bar_chart_with_ci(
                    summary_dict, metric,
                    save_path=os.path.join(chart_dir, f"bar_{metric}.png"),
                )
                if path:
                    chart_files.append(path)

        return chart_files

    def _save_experiment(self, result: ExperimentResult, status: str = "completed"):
        """保存实验到数据库"""
        try:
            self.db.save_experiment(result)
        except Exception as e:
            logger.warning(f"保存实验到数据库失败: {e}")

    def _save_benchmark_scores(self, experiment_id: str, benchmark_result):
        """保存benchmark得分到数据库"""
        try:
            for algo_name, algo_result in benchmark_result.algorithms.items():
                self.db.save_algorithm_scores(
                    experiment_id, algo_name, algo_result.metrics
                )
        except Exception as e:
            logger.warning(f"保存算法得分失败: {e}")

    def _get_git_hash(self) -> str:
        """获取当前git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _update_progress(self, callback, stage: str, progress: int, total: int):
        """更新进度"""
        if callback:
            callback(stage, progress, total)
        logger.info(f"  [{progress}/{total}] {stage}...")


# ===========================
# 便捷函数
# ===========================


def run_quick_experiment(
    algorithms: Optional[List[str]] = None,
    dataset_key: str = "builtin",
    n_samples: int = 20,
) -> ExperimentResult:
    """
    快速实验便捷函数

    Args:
        algorithms: 算法列表（None=所有已注册算法）
        dataset_key: 数据集键名
        n_samples: 样本数

    Returns:
        ExperimentResult
    """
    if algorithms is None:
        algorithms = DenoiserRegistry.list_denoisers()

    config = ExperimentConfig(
        name=f"快速测评 - {dataset_key}",
        description=f"自动生成的快速测评实验，{len(algorithms)}个算法, {n_samples}个样本",
        algorithms=algorithms,
        dataset_key=dataset_key,
        dataset_config={"n_samples": n_samples},
        significance_test=True,
        visualization=True,
        efficiency_profile=False,
    )

    runner = ExperimentRunner()
    return runner.run(config)


def reproduce_experiment(experiment_id: str, db_path: str = "./data/experiments/results.db") -> Optional[ExperimentResult]:
    """
    复现已有实验

    Args:
        experiment_id: 要复现的实验ID
        db_path: 数据库路径

    Returns:
        ExperimentResult 或 None
    """
    db = ExperimentDB(db_path)
    exp = db.get_experiment(experiment_id)

    if not exp:
        logger.error(f"实验不存在: {experiment_id}")
        return None

    config_dict = exp.get("config_json", {})
    if isinstance(config_dict, str):
        config_dict = json.loads(config_dict)

    config = ExperimentConfig(**config_dict)
    config.name = f"{config.name} (复现)"

    runner = ExperimentRunner(db_path=db_path)
    return runner.run(config)
