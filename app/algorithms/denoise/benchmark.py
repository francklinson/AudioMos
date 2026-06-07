"""
降噪算法标准测评基准

提供统一的算法测评框架，支持:
- 多算法对比评测
- 标准测试数据集
- 可重复的评测流程
- 结果汇总与报告生成

使用方式:
    runner = BenchmarkRunner()
    results = runner.run_benchmark(
        algorithms=["speechbrain_metricgan", "spectral_subtraction"],
        test_files=["path/to/noisy1.wav"],
        reference_files=["path/to/clean1.wav"]
    )
    runner.print_summary(results)
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import soundfile as sf

from .base import BaseDenoiser, DenoiseResult
from .registry import DenoiserRegistry
from .evaluator import DenoiseMetrics, DenoiseEvaluation, DenoiseEvaluator
from .dataset_dns import DNSDataset, DNSMixConfig

logger = logging.getLogger(__name__)


# ===========================
# 数据类
# ===========================


@dataclass
class AlgorithmBenchmarkResult:
    """单个算法的测评结果"""

    algorithm_name: str
    n_files: int
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # metric_name -> {mean, std, median, min, max}
    rtf: float = 0.0
    avg_processing_time: float = 0.0
    details: List[DenoiseEvaluation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """完整测评结果"""

    timestamp: str
    dataset_info: Dict
    algorithms: Dict[str, AlgorithmBenchmarkResult] = field(default_factory=dict)
    ranking: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)  # metric -> [(algo, score)]


# ===========================
# 测评运行器
# ===========================


class BenchmarkRunner:
    """
    降噪算法测评运行器

    统一接口进行多算法对比测评。
    """

    def __init__(self, output_dir: str = "./data/benchmark_results", device: str = "cuda"):
        """
        初始化测评运行器

        Args:
            output_dir: 结果输出目录
            device: 计算设备
        """
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

        # 初始化评估器
        try:
            self.evaluator = DenoiseEvaluator(device=device)
        except Exception as e:
            logger.warning(f"评估器初始化失败（部分指标不可用）: {e}")
            self.evaluator = None

    def run_benchmark(
        self,
        algorithms: List[str],
        test_files: List[str],
        reference_files: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> BenchmarkResult:
        """
        运行标准测评

        Args:
            algorithms: 算法名称列表
            test_files: 测试音频文件列表（带噪）
            reference_files: 参考音频文件列表（干净，可选）
            progress_callback: 进度回调

        Returns:
            BenchmarkResult 对象
        """
        result = BenchmarkResult(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset_info={
                "n_files": len(test_files),
                "has_reference": reference_files is not None and len(reference_files) > 0,
            },
        )

        total_tasks = len(algorithms) * len(test_files)
        completed = 0

        for algo_name in algorithms:
            algo_result = AlgorithmBenchmarkResult(algorithm_name=algo_name, n_files=len(test_files))

            # 初始化算法
            denoiser = DenoiserRegistry.get(algo_name, device=self.device)
            if denoiser is None:
                algo_result.errors.append(f"无法加载算法: {algo_name}")
                result.algorithms[algo_name] = algo_result
                continue

            try:
                denoiser.initialize()
            except Exception as e:
                algo_result.errors.append(f"算法初始化失败: {e}")
                result.algorithms[algo_name] = algo_result
                continue

            # 处理所有文件
            evaluations = []
            processing_times = []

            for i, test_file in enumerate(test_files):
                try:
                    # 执行降噪
                    output_path = os.path.join(self.output_dir, f"{algo_name}_output_{i}.wav")
                    process_result = denoiser.denoise_file(test_file, output_path)

                    # 评估 (evaluate_file返回DenoiseMetrics)
                    if reference_files and i < len(reference_files):
                        ev_metrics = self.evaluator.evaluate_file(
                            output_path,
                            reference_files[i],
                            process_result.processing_time,
                        )
                    else:
                        ev_metrics = self.evaluator.evaluate_without_reference(
                            process_result.audio, process_result.processing_time
                        )

                    evaluations.append(ev_metrics)
                    processing_times.append(process_result.processing_time)

                except Exception as e:
                    algo_result.errors.append(f"处理失败 {test_file}: {e}")
                    logger.warning(f"处理失败: {e}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_tasks, algo_name)

            # 计算汇总统计
            algo_result.details = evaluations
            algo_result.avg_processing_time = np.mean(processing_times) if processing_times else 0
            algo_result.metrics = self._compute_aggregate_metrics(evaluations)
            algo_result.rtf = (
                0 if not processing_times else np.mean(processing_times)
            )

            result.algorithms[algo_name] = algo_result

        # 计算排名
        result.ranking = self._compute_ranking(result)

        return result

    def _compute_aggregate_metrics(self, evaluations) -> Dict[str, Dict[str, float]]:
        """计算聚合指标统计 (接受DenoiseMetrics或DenoiseEvaluation列表)"""
        if not evaluations:
            return {}

        # 收集各指标值
        metric_values = defaultdict(list)

        for ev in evaluations:
            # 支持DenoiseMetrics和DenoiseEvaluation两种类型
            if hasattr(ev, 'metrics') and ev.metrics is not None:
                m = ev.metrics
            else:
                m = ev  # DenoiseMetrics本身就是指标对象

            for key in ["pesq", "stoi", "sisdr", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"]:
                val = getattr(m, key, None)
                if val is not None and not np.isnan(val) and not np.isinf(val):
                    metric_values[key].append(val)

            proc_time = getattr(m, 'processing_time', 0) or getattr(ev, 'processing_time', 0)
            if proc_time:
                metric_values["processing_time"].append(proc_time)

        # 计算统计量
        stats = {}
        for metric_name, values in metric_values.items():
            if values:
                arr = np.array(values)
                stats[metric_name] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "std": round(float(np.std(arr)), 4),
                    "median": round(float(np.median(arr)), 4),
                    "min": round(float(np.min(arr)), 4),
                    "max": round(float(np.max(arr)), 4),
                }

        return stats

    def _compute_ranking(self, result: BenchmarkResult) -> Dict[str, List[Tuple[str, float]]]:
        """计算各指标的算法排名"""
        rankings = {}

        # 需要排序的指标（更高更好）
        higher_better = ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]
        # 需要排序的指标（更低更好）
        lower_better = ["processing_time"]

        for metric in higher_better + lower_better:
            algo_scores = []
            for algo_name, algo_result in result.algorithms.items():
                if metric in algo_result.metrics:
                    score = algo_result.metrics[metric].get("mean", 0)
                    algo_scores.append((algo_name, score))

            if algo_scores:
                reverse = metric in higher_better
                algo_scores.sort(key=lambda x: x[1], reverse=reverse)
                rankings[metric] = algo_scores

        return rankings

    def run_on_simple_dataset(self, algorithms: List[str]) -> BenchmarkResult:
        """
        使用项目内置测试数据运行快速测评

        Args:
            algorithms: 算法列表

        Returns:
            BenchmarkResult
        """
        dataset = DNSDataset()
        test_files, ref_files = dataset.prepare_simple_test_set()

        if not test_files:
            print("未找到测试文件，使用合成数据")
            # 生成简单的合成测试数据
            test_files, ref_files = self._generate_synthetic_test_set()

        return self.run_benchmark(algorithms, test_files, ref_files)

    def _generate_synthetic_test_set(self, n_files: int = 10) -> Tuple[List[str], List[str]]:
        """生成合成测试数据集"""
        test_dir = os.path.join(self.output_dir, "synthetic_test")
        os.makedirs(test_dir, exist_ok=True)

        test_files = []
        ref_files = []

        for i in range(n_files):
            sr = 16000
            duration = np.random.uniform(2.0, 5.0)
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)

            # 生成简单的合成语音（多个正弦波叠加模拟语音）
            clean = (
                0.3 * np.sin(2 * np.pi * 200 * t)
                + 0.2 * np.sin(2 * np.pi * 500 * t)
                + 0.15 * np.sin(2 * np.pi * 1000 * t)
                + 0.1 * np.sin(2 * np.pi * 2000 * t)
            )

            # 添加噪声
            noise_type = np.random.choice(["white", "pink", "babble"])
            if noise_type == "white":
                noise = np.random.randn(len(t)) * 0.1
            elif noise_type == "pink":
                # 粉红噪声模拟
                noise = np.cumsum(np.random.randn(len(t))) * 0.001
            else:
                # babble-like噪声
                noise = 0.1 * np.sin(2 * np.pi * 100 * t) * np.random.randn(len(t))

            noisy = clean + noise * np.random.uniform(0.5, 2.0)

            clean_path = os.path.join(test_dir, f"synthetic_clean_{i:04d}.wav")
            noisy_path = os.path.join(test_dir, f"synthetic_noisy_{i:04d}.wav")

            sf.write(clean_path, clean, sr)
            sf.write(noisy_path, noisy, sr)

            test_files.append(noisy_path)
            ref_files.append(clean_path)

        return test_files, ref_files

    def print_summary(self, result: BenchmarkResult):
        """打印测评结果摘要"""
        print("\n" + "=" * 80)
        print(f"  降噪算法测评结果")
        print(f"  时间: {result.timestamp}")
        print(f"  文件数: {result.dataset_info.get('n_files', 0)}")
        print("=" * 80)

        # 算法汇总表
        print(f"\n{'算法':<30} {'PESQ':>8} {'STOI':>8} {'SI-SDR':>8} {'DNSMOS':>8} {'耗时(s)':>8}")
        print("-" * 75)

        for algo_name, algo_result in result.algorithms.items():
            metrics = algo_result.metrics
            pesq = metrics.get("pesq", {}).get("mean", "-")
            stoi = metrics.get("stoi", {}).get("mean", "-")
            sisdr = metrics.get("sisdr", {}).get("mean", "-")
            dnsmos = metrics.get("dnsmos_ovrl", {}).get("mean", "-")
            proc_time = metrics.get("processing_time", {}).get("mean", "-")

            pesq_str = f"{pesq:.2f}" if isinstance(pesq, (int, float)) else str(pesq)
            stoi_str = f"{stoi:.3f}" if isinstance(stoi, (int, float)) else str(stoi)
            sisdr_str = f"{sisdr:.1f}" if isinstance(sisdr, (int, float)) else str(sisdr)
            dnsmos_str = f"{dnsmos:.2f}" if isinstance(dnsmos, (int, float)) else str(dnsmos)
            time_str = f"{proc_time:.2f}" if isinstance(proc_time, (int, float)) else str(proc_time)

            print(f"{algo_name:<30} {pesq_str:>8} {stoi_str:>8} {sisdr_str:>8} {dnsmos_str:>8} {time_str:>8}")

        # 排名
        if result.ranking:
            print(f"\n{'指标':<15} {'第1名':<25} {'第2名':<25} {'第3名':<25}")
            print("-" * 80)
            for metric, rank_list in result.ranking.items():
                if rank_list:
                    metric_name = {
                        "pesq": "PESQ",
                        "stoi": "STOI",
                        "sisdr": "SI-SDR",
                        "dnsmos_ovrl": "DNSMOS",
                        "processing_time": "速度",
                    }.get(metric, metric)

                    row = f"{metric_name:<15}"
                    for i in range(min(3, len(rank_list))):
                        algo, score = rank_list[i]
                        if isinstance(score, float):
                            row += f" {algo}({score:.2f})"
                        else:
                            row += f" {algo}"
                    print(row)

        print("=" * 80)

    def run_with_statistics(
        self,
        algorithms: List[str],
        test_files: List[str],
        reference_files: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[BenchmarkResult, Optional[any]]:
        """
        运行带统计显著性检验的测评

        Args:
            algorithms: 算法列表
            test_files: 测试文件列表
            reference_files: 参考文件列表
            progress_callback: 进度回调

        Returns:
            (BenchmarkResult, 统计报告或None)
        """
        from .significance import StatisticalAnalyzer

        # 运行基础测评
        benchmark_result = self.run_benchmark(algorithms, test_files, reference_files, progress_callback)

        if len(algorithms) < 2:
            logger.warning("至少需要2个算法才能进行统计检验")
            return benchmark_result, None

        # 统计检验
        analyzer = StatisticalAnalyzer()
        algo_scores = {}

        for algo_name, algo_result in benchmark_result.algorithms.items():
            scores = {}
            for metric, stats in algo_result.metrics.items():
                metric_values = []
                for detail in algo_result.details:
                    if hasattr(detail, 'metrics'):
                        val = getattr(detail.metrics, metric, None)
                    else:
                        val = getattr(detail, metric, None)
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        metric_values.append(val)
                if metric_values:
                    scores[metric] = metric_values
            if scores:
                algo_scores[algo_name] = scores

        comparison = None
        if len(algo_scores) >= 2:
            algo_names = sorted(algo_scores.keys())
            comparison = analyzer.compare_algorithms(
                algo_scores[algo_names[0]],
                algo_scores[algo_names[1]],
                name_a=algo_names[0],
                name_b=algo_names[1],
            )

        # 保存统计报告
        if comparison:
            stats_path = os.path.join(self.output_dir, f"statistical_report_{int(time.time())}.json")
            with open(stats_path, "w") as f:
                json.dump(comparison.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"统计报告已保存: {stats_path}")

        return benchmark_result, comparison

    def run_with_visualization(
        self, result: BenchmarkResult, chart_dir: Optional[str] = None
    ) -> List[str]:
        """
        为测评结果生成可视化图表

        Args:
            result: BenchmarkResult 对象
            chart_dir: 图表输出目录

        Returns:
            图表文件路径列表
        """
        from .visualizer import BenchmarkVisualizer

        if chart_dir is None:
            chart_dir = os.path.join(self.output_dir, "charts")

        viz = BenchmarkVisualizer(output_dir=chart_dir)
        chart_files = []

        # 提取得分和汇总
        algo_scores = {}
        algo_means = {}
        for algo_name, algo_result in result.algorithms.items():
            algo_means[algo_name] = {}
            for metric, stats in algo_result.metrics.items():
                algo_means[algo_name][metric] = stats.get("mean", 0)
                if metric not in algo_scores:
                    algo_scores[metric] = {}
                algo_scores[metric][algo_name] = stats.get("mean", 0)

        # 雷达图
        if len(algo_means) > 1:
            path = viz.radar_chart(algo_means)
            if path:
                chart_files.append(path)

        # 各指标箱线图
        for metric in ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]:
            scores_dict = {}
            for algo_name, algo_result in result.algorithms.items():
                values = []
                for detail in algo_result.details:
                    if hasattr(detail, 'metrics'):
                        val = getattr(detail.metrics, metric, None)
                    else:
                        val = getattr(detail, metric, None)
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        values.append(val)
                if values:
                    scores_dict[algo_name] = values

            if len(scores_dict) > 1:
                path = viz.box_plot(scores_dict, metric)
                if path:
                    chart_files.append(path)

        # 柱状图
        for metric in ["pesq", "stoi", "sisdr"]:
            summary_dict = {}
            for algo_name, algo_result in result.algorithms.items():
                if metric in algo_result.metrics:
                    summary_dict[algo_name] = {metric: algo_result.metrics[metric]}
            if len(summary_dict) > 1:
                path = viz.bar_chart_with_ci(summary_dict, metric)
                if path:
                    chart_files.append(path)

        # 仪表盘
        dashboard_path = viz.generate_dashboard_html(
            algo_means, chart_files=chart_files,
            save_path=os.path.join(chart_dir, "dashboard.html"),
        )
        if dashboard_path:
            chart_files.append(dashboard_path)

        logger.info(f"生成 {len(chart_files)} 个可视化图表")
        return chart_files

    def run_full_benchmark(
        self,
        algorithms: List[str],
        test_files: List[str],
        reference_files: Optional[List[str]] = None,
        enable_statistics: bool = True,
        enable_visualization: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        运行完整测评流程（含统计+可视化）

        Args:
            algorithms: 算法列表
            test_files: 测试文件列表
            reference_files: 参考文件列表
            enable_statistics: 是否启用统计检验
            enable_visualization: 是否启用可视化
            progress_callback: 进度回调

        Returns:
            包含 benchmark_result, stats, charts 的字典
        """
        if enable_statistics:
            benchmark_result, stats = self.run_with_statistics(
                algorithms, test_files, reference_files, progress_callback
            )
        else:
            benchmark_result = self.run_benchmark(algorithms, test_files, reference_files, progress_callback)
            stats = None

        charts = []
        if enable_visualization:
            charts = self.run_with_visualization(benchmark_result)

        # 保存完整结果
        json_path = self.export_results(benchmark_result)

        return {
            "benchmark_result": benchmark_result,
            "statistical_report": stats,
            "chart_files": charts,
            "json_export": json_path,
        }

    def analyze_robustness(
        self,
        benchmark_result: BenchmarkResult,
        sample_metadata: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        """
        鲁棒性分析: 按噪声类型和SNR级别分组分析

        Args:
            benchmark_result: 基准测试结果
            sample_metadata: 样本元数据 {file_index: {noise_type, snr_db}}

        Returns:
            鲁棒性分析报告
        """
        report = {
            "by_noise_type": {},
            "by_snr": {},
            "summary": "",
        }

        if not sample_metadata:
            report["summary"] = "需要提供样本元数据以进行鲁棒性分析"
            return report

        # 从详细信息中提取指标并按组分类
        for algo_name, algo_result in benchmark_result.algorithms.items():
            noise_groups = defaultdict(list)
            snr_groups = defaultdict(list)

            for i, detail in enumerate(algo_result.details):
                metadata = sample_metadata.get(str(i), sample_metadata.get(i, {}))
                noise_type = metadata.get("noise_type", "unknown")
                snr_db = metadata.get("snr_db", None)

                metrics_obj = detail.metrics if hasattr(detail, 'metrics') else detail

                for metric in ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]:
                    val = getattr(metrics_obj, metric, None)
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        noise_groups[f"{metric}_{noise_type}"].append(val)
                        if snr_db is not None:
                            snr_groups[f"{metric}_{snr_db}dB"].append(val)

            # 汇总噪声类型
            for group_key, values in noise_groups.items():
                if values:
                    report["by_noise_type"][f"{algo_name}/{group_key}"] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std": round(float(np.std(values)), 4),
                        "n": len(values),
                    }

            # 汇总SNR
            for group_key, values in snr_groups.items():
                if values:
                    report["by_snr"][f"{algo_name}/{group_key}"] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std": round(float(np.std(values)), 4),
                        "n": len(values),
                    }

        # 生成摘要
        lines = ["鲁棒性分析摘要"]
        if report["by_noise_type"]:
            lines.append(f"  噪声类型分组数: {len(set(k.split('/')[0] for k in report['by_noise_type']))}")
        if report["by_snr"]:
            lines.append(f"  SNR级别分组数: {len(set(k.split('/')[0] for k in report['by_snr']))}")
        report["summary"] = "\n".join(lines)

        return report

    def export_results(self, result: BenchmarkResult, output_file: Optional[str] = None) -> str:
        """
        导出测评结果为JSON

        Args:
            result: 测评结果
            output_file: 输出文件路径

        Returns:
            输出文件路径
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"benchmark_{timestamp}.json")

        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (DenoiseMetrics, DenoiseEvaluation)):
                return obj.__dict__ if hasattr(obj, "__dict__") else str(obj)
            return str(obj)

        serializable = json.loads(json.dumps(result, default=convert_to_serializable))

        with open(output_file, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        return output_file


def run_quick_benchmark(algorithms: Optional[List[str]] = None) -> BenchmarkResult:
    """
    快速测评便捷函数

    Args:
        algorithms: 算法列表，默认使用所有已注册算法

    Returns:
        BenchmarkResult
    """
    if algorithms is None:
        algorithms = DenoiserRegistry.list_denoisers()

    runner = BenchmarkRunner()
    result = runner.run_on_simple_dataset(algorithms)
    runner.print_summary(result)

    return result


if __name__ == "__main__":
    # 快速测试
    result = run_quick_benchmark()
