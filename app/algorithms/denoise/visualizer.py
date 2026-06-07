"""
降噪算法测评可视化模块

基于matplotlib + seaborn生成专业的测评图表。
支持:
- 雷达图: 多指标综合对比
- 箱线图: 指标分布对比
- 柱状图(带置信区间): 均值对比
- 热力图: 显著性矩阵/相关矩阵
- 散点图: 指标间关系
- 仪表盘: HTML综合展示
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ===========================
# 配置
# ===========================

# 中文字体设置
try:
    import matplotlib
    # 尝试使用中文字体
    _chinese_fonts = [
        'DejaVu Sans', 'sans-serif',
    ]
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = _chinese_fonts + matplotlib.rcParams.get('font.sans-serif', [])
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
except ImportError:
    SEABORN_AVAILABLE = False

# 指标显示名称和颜色
METRIC_DISPLAY = {
    "pesq": ("PESQ", "#2196F3"),
    "stoi": ("STOI", "#4CAF50"),
    "sisdr": ("SI-SDR (dB)", "#FF9800"),
    "dnsmos_ovrl": ("DNSMOS OVRL", "#9C27B0"),
    "dnsmos_sig": ("DNSMOS SIG", "#00BCD4"),
    "dnsmos_bak": ("DNSMOS BAK", "#E91E63"),
    "nisqa_mos": ("NISQA MOS", "#795548"),
    "utmos": ("UTMOS", "#3F51B5"),
    "processing_time": ("处理时间 (s)", "#F44336"),
    "rtf": ("RTF", "#607D8B"),
}


DEFAULT_CHART_DPI = 150
DEFAULT_FIGSIZE = (10, 6)


# ===========================
# 可视化器
# ===========================


class BenchmarkVisualizer:
    """
    降噪测评结果可视化器

    使用方式:
        viz = BenchmarkVisualizer(output_dir="./results/charts")
        viz.radar_chart(benchmark_result, save_path="radar.png")
        viz.box_plot(benchmark_result, metric="pesq")
        viz.generate_dashboard_html(benchmark_result, stats_report)
    """

    def __init__(self, output_dir: str = "./data/benchmark_results/charts"):
        """
        初始化可视化器

        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def radar_chart(
        self,
        algorithm_metrics: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        title: str = "降噪算法多指标综合对比",
        figsize: Tuple[int, int] = (10, 8),
    ) -> str:
        """
        雷达图: 多算法多指标综合对比

        Args:
            algorithm_metrics: {algo_name: {metric_name: value}}
            save_path: 保存路径
            title: 图表标题
            figsize: 图表尺寸

        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = str(self.output_dir / f"radar_{int(time.time())}.png")

        # 确定要展示的指标（所有算法共有的）
        all_metrics = set()
        for scores in algorithm_metrics.values():
            all_metrics.update(scores.keys())

        # 筛选可用的指标（排除效率类）
        quality_metrics = [
            m for m in ["pesq", "stoi", "sisdr", "dnsmos_ovrl", "nisqa_mos", "utmos"]
            if m in all_metrics
        ]

        if len(quality_metrics) < 2:
            logger.warning("指标不足，跳过雷达图")
            return ""

        # 归一化到[0,1]
        normalized = {}
        for algo, scores in algorithm_metrics.items():
            normalized[algo] = []
            for m in quality_metrics:
                val = scores.get(m, None)
                if val is None or np.isnan(val) or np.isinf(val):
                    normalized[algo].append(0)
                    continue

                # 各指标的归一化范围
                if m == "pesq":
                    norm_val = max(0, min(1, val / 4.5))
                elif m == "stoi":
                    norm_val = max(0, min(1, val))
                elif m == "sisdr":
                    norm_val = max(0, min(1, val / 20))
                elif m == "dnsmos_ovrl":
                    norm_val = max(0, min(1, val / 5))
                elif m == "nisqa_mos":
                    norm_val = max(0, min(1, val / 5))
                elif m == "utmos":
                    norm_val = max(0, min(1, val / 5))
                else:
                    norm_val = max(0, min(1, val / 5))
                normalized[algo].append(norm_val)

        # 绘制雷达图
        n_metrics = len(quality_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 标签
        labels = [METRIC_DISPLAY.get(m, (m,))[0] for m in quality_metrics]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithm_metrics)))

        for idx, (algo, values) in enumerate(normalized.items()):
            values_closed = values + values[:1]
            color = colors[idx % len(colors)]
            ax.fill(angles, values_closed, alpha=0.1, color=color)
            ax.plot(angles, values_closed, 'o-', linewidth=2, label=algo, color=color, markersize=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=DEFAULT_CHART_DPI, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"雷达图已保存: {save_path}")
        return save_path

    def box_plot(
        self,
        scores_dict: Dict[str, List[float]],
        metric: str,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
        """
        箱线图: 各算法在某指标上的分布对比

        Args:
            scores_dict: {algo_name: [scores]}
            metric: 指标名称
            save_path: 保存路径
            title: 标题
            figsize: 尺寸

        Returns:
            保存路径
        """
        if save_path is None:
            save_path = str(self.output_dir / f"box_{metric}_{int(time.time())}.png")

        # 过滤有效数据
        plot_data = {}
        for algo, scores in scores_dict.items():
            valid = [s for s in scores if s is not None and not np.isnan(s) and not np.isinf(s)]
            if valid:
                plot_data[algo] = valid

        if not plot_data:
            logger.warning(f"无有效数据用于 {metric} 箱线图")
            return ""

        fig, ax = plt.subplots(figsize=figsize)

        labels = list(plot_data.keys())
        data = list(plot_data.values())

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

        # 着色
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        metric_name, _ = METRIC_DISPLAY.get(metric, (metric,))
        if title is None:
            title = f"{metric_name} 分布对比"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xlabel("算法", fontsize=11)
        ax.tick_params(axis='x', rotation=30)

        # 添加均值标注
        for i, scores in enumerate(data):
            mean_val = np.mean(scores)
            ax.annotate(f'{mean_val:.3f}', xy=(i + 1, mean_val),
                       xytext=(i + 1.3, mean_val), fontsize=8,
                       arrowprops=dict(arrowstyle='->', color='red'),
                       color='red', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=DEFAULT_CHART_DPI, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"箱线图已保存: {save_path}")
        return save_path

    def bar_chart_with_ci(
        self,
        summary_dict: Dict[str, Dict[str, Dict[str, float]]],
        metric: str,
        save_path: Optional[str] = None,
        ci_data: Optional[Dict[str, Tuple[float, float]]] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
        """
        带误差条的柱状图

        Args:
            summary_dict: {algo: {metric: {mean, std}}}
            metric: 指标名称
            save_path: 保存路径
            ci_data: {algo: (lower, upper)} 置信区间
            title: 标题
            figsize: 尺寸

        Returns:
            保存路径
        """
        if save_path is None:
            save_path = str(self.output_dir / f"bar_{metric}_{int(time.time())}.png")

        algos = []
        means = []
        errors = []

        for algo in sorted(summary_dict.keys()):
            if metric in summary_dict[algo]:
                stats = summary_dict[algo][metric]
                mean_val = stats.get("mean", 0)
                if mean_val is None or np.isnan(mean_val) or np.isinf(mean_val):
                    continue
                algos.append(algo)
                means.append(mean_val)

                if ci_data and algo in ci_data:
                    lower, upper = ci_data[algo]
                    errors.append((mean_val - lower, upper - mean_val))
                else:
                    std_val = stats.get("std", 0) or 0
                    errors.append(std_val)

        if not algos:
            logger.warning(f"无有效数据用于 {metric} 柱状图")
            return ""

        fig, ax = plt.subplots(figsize=figsize)

        x = range(len(algos))
        colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

        # 准备误差数据
        if errors and isinstance(errors[0], tuple):
            yerr = list(zip(*errors))
        else:
            yerr = errors

        bars = ax.bar(x, means, yerr=yerr, capsize=5, color=colors, alpha=0.8, edgecolor='white')

        metric_name, _ = METRIC_DISPLAY.get(metric, (metric,))
        if title is None:
            title = f"{metric_name} 均值对比 (95% CI)"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # 数值标注
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=DEFAULT_CHART_DPI, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"柱状图已保存: {save_path}")
        return save_path

    def significance_heatmap(
        self,
        p_value_matrix: Dict[Tuple[str, str], float],
        algorithms: List[str],
        metric: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> str:
        """
        统计显著性热力图

        Args:
            p_value_matrix: {(algo_a, algo_b): p_value}
            algorithms: 算法名称列表
            metric: 指标名称
            save_path: 保存路径
            figsize: 尺寸

        Returns:
            保存路径
        """
        if save_path is None:
            save_path = str(self.output_dir / f"heatmap_{metric}_{int(time.time())}.png")

        n = len(algorithms)
        p_matrix = np.ones((n, n))

        for i, a1 in enumerate(algorithms):
            for j, a2 in enumerate(algorithms):
                if i == j:
                    p_matrix[i, j] = 1.0
                elif (a1, a2) in p_value_matrix:
                    p_matrix[i, j] = p_value_matrix[(a1, a2)]
                elif (a2, a1) in p_value_matrix:
                    p_matrix[i, j] = p_value_matrix[(a2, a1)]

        fig, ax = plt.subplots(figsize=figsize)

        # 对角线下三角掩码
        mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)

        cmap = sns.diverging_palette(240, 10, as_cmap=True) if SEABORN_AVAILABLE else plt.cm.RdYlGn_r

        if SEABORN_AVAILABLE:
            sns.heatmap(
                p_matrix, mask=mask, annot=True, fmt='.4f',
                cmap=cmap, vmin=0, vmax=1,
                xticklabels=algorithms, yticklabels=algorithms,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'p-value'},
            )
        else:
            im = ax.imshow(p_matrix, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='p-value')
            ax.set_xticks(range(n))
            ax.set_xticklabels(algorithms, rotation=45, ha='right')
            ax.set_yticks(range(n))
            ax.set_yticklabels(algorithms)
            for i in range(n):
                for j in range(n):
                    if not mask[i, j]:
                        ax.text(j, i, f'{p_matrix[i, j]:.4f}', ha='center', va='center', fontsize=9)

        metric_name, _ = METRIC_DISPLAY.get(metric, (metric,))
        ax.set_title(f"{metric_name} - 显著性矩阵 (p值)", fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=DEFAULT_CHART_DPI, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"热力图已保存: {save_path}")
        return save_path

    def scatter_comparison(
        self,
        scores_a: List[float],
        scores_b: List[float],
        metric: str,
        name_a: str = "Algorithm A",
        name_b: str = "Algorithm B",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
    ) -> str:
        """
        散点图: 直观展示两个算法在每个样本上的得分关系

        Args:
            scores_a, scores_b: 两个算法的得分
            metric: 指标名称
            name_a, name_b: 算法名称
            save_path: 保存路径
            figsize: 尺寸

        Returns:
            保存路径
        """
        if save_path is None:
            save_path = str(self.output_dir / f"scatter_{metric}_{int(time.time())}.png")

        sa = np.array([s for s in scores_a if s is not None and not np.isnan(s) and not np.isinf(s)])
        sb = np.array([s for s in scores_b if s is not None and not np.isnan(s) and not np.isinf(s)])
        min_len = min(len(sa), len(sb))
        sa = sa[:min_len]
        sb = sb[:min_len]

        if len(sa) < 2:
            logger.warning(f"数据不足，跳过 {metric} 散点图")
            return ""

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(sa, sb, alpha=0.6, c='#2196F3', edgecolors='white', linewidths=0.5)

        # 对角线（y=x参考线）
        all_vals = np.concatenate([sa, sb])
        min_val, max_val = np.min(all_vals), np.max(all_vals)
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
               [min_val - margin, max_val + margin],
               '--', color='red', alpha=0.5, linewidth=1.5, label='y=x (相等)')

        ax.set_xlabel(f"{name_a}", fontsize=11)
        ax.set_ylabel(f"{name_b}", fontsize=11)
        metric_name, _ = METRIC_DISPLAY.get(metric, (metric,))
        ax.set_title(f"{metric_name} - {name_a} vs {name_b}", fontsize=14, fontweight='bold')

        # 标注高于/低于对角线的点数
        above = np.sum(sb > sa)
        below = np.sum(sb < sa)
        ax.text(0.02, 0.98, f"{name_b} 更优: {above}/{len(sa)}\n{name_a} 更优: {below}/{len(sa)}",
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(save_path, dpi=DEFAULT_CHART_DPI, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"散点图已保存: {save_path}")
        return save_path

    def generate_dashboard_html(
        self,
        algorithm_metrics: Dict[str, Dict[str, Dict[str, float]]],
        chart_files: Optional[List[str]] = None,
        title: str = "降噪算法测评仪表盘",
        save_path: Optional[str] = None,
    ) -> str:
        """
        生成HTML格式的综合仪表盘

        将多个图表嵌入一个HTML页面。

        Args:
            algorithm_metrics: {algo: {metric: {mean, std, ...}}}
            chart_files: 已生成的图表文件路径列表
            title: 仪表盘标题
            save_path: 保存路径

        Returns:
            保存路径
        """
        if save_path is None:
            save_path = str(self.output_dir / f"dashboard_{int(time.time())}.html")

        # 生成汇总表
        summary_rows = ""
        for algo, metrics in algorithm_metrics.items():
            row = f"<tr><td><strong>{algo}</strong></td>"
            for metric in ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]:
                mv = metrics.get(metric, None)
                # 兼容两种格式: float值 或 {"mean": ..., ...}
                if isinstance(mv, dict):
                    val = mv.get("mean", None)
                else:
                    val = mv
                if val is not None and not (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    row += f"<td>{val:.3f}</td>"
                else:
                    row += "<td>N/A</td>"
            processing_time = metrics.get("processing_time", None)
            if isinstance(processing_time, dict):
                processing_time = processing_time.get("mean", None)
            if processing_time is not None:
                row += f"<td>{processing_time:.3f}</td>"
            else:
                row += "<td>N/A</td>"
            row += "</tr>"
            summary_rows += row

        # 图表嵌入
        chart_imgs = ""
        if chart_files:
            for chart_path in chart_files:
                if chart_path and os.path.exists(chart_path):
                    # 使用相对路径
                    rel_path = os.path.relpath(chart_path, os.path.dirname(save_path))
                    chart_name = os.path.basename(chart_path).split('_')[0]
                    chart_imgs += f"""
                    <div style="margin:20px 0; text-align:center;">
                        <h3>{chart_name}</h3>
                        <img src="{rel_path}" alt="{chart_name}" style="max-width:100%; border:1px solid #ddd; border-radius:4px;">
                    </div>
                    """

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 28px; margin-bottom: 5px; }}
        .header p {{ opacity: 0.9; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
        .section h2 {{ font-size: 20px; margin-bottom: 15px; color: #555; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #667eea; color: white; font-weight: 500; }}
        tr:hover {{ background: #f8f9ff; }}
        .metric-badge {{ display: inline-block; padding: 4px 10px; background: #e3f2fd; border-radius: 4px; margin: 3px; font-size: 13px; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <p>生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="section">
            <h2>📋 算法性能汇总</h2>
            <table>
                <thead>
                    <tr>
                        <th>算法</th>
                        <th>PESQ</th>
                        <th>STOI</th>
                        <th>SI-SDR (dB)</th>
                        <th>DNSMOS OVRL</th>
                        <th>处理时间 (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_rows}
                </tbody>
            </table>
            <div style="margin-top: 15px;">
                <span class="metric-badge">PESQ: 感知语音质量 (1-4.5)</span>
                <span class="metric-badge">STOI: 短时客观可懂度 (0-1)</span>
                <span class="metric-badge">SI-SDR: 尺度不变信噪比 (dB)</span>
                <span class="metric-badge">DNSMOS: 深度降噪MOS (1-5)</span>
            </div>
        </div>

        <div class="section">
            <h2>📈 可视化图表</h2>
            {chart_imgs if chart_imgs else "<p>暂无图表（运行完整测评以生成）</p>"}
        </div>

        <div class="footer">
            <p>AudioMOS 降噪算法测评系统 | 自动生成</p>
        </div>
    </div>
</body>
</html>"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"仪表盘已保存: {save_path}")
        return save_path
