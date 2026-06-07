"""
降噪测评报告生成模块
生成详细的测评报告，包括图表和对比分析
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import time
from datetime import datetime

from .evaluator import DenoiseEvaluation


class ReportGenerator:
    """
    降噪测评报告生成器
    支持生成多种格式的报告
    """
    
    def __init__(self, output_dir: str = "./data/denoise_reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_excel_report(
        self,
        results: Dict[str, List[DenoiseEvaluation]],
        output_file: Optional[str] = None
    ) -> str:
        """
        生成Excel格式的详细报告
        
        Args:
            results: 测评结果字典
            output_file: 输出文件路径(可选)
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"denoise_evaluation_{timestamp}.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. 详细结果表
            all_data = []
            for algo_name, evaluations in results.items():
                for eval in evaluations:
                    row = {
                        '算法名称': algo_name,
                        '文件名': eval.file_name,
                        'PESQ': eval.metrics.pesq,
                        'STOI': eval.metrics.stoi,
                        'SI-SDR (dB)': eval.metrics.sisdr,
                        'DNSMOS OVRL': eval.metrics.dnsmos_ovrl,
                        'DNSMOS SIG': eval.metrics.dnsmos_sig,
                        'DNSMOS BAK': eval.metrics.dnsmos_bak,
                        'NISQA MOS': eval.metrics.nisqa_mos,
                        'UTMOS': eval.metrics.utmos,
                        '处理时间(s)': eval.metrics.processing_time,
                        '实时因子(RTF)': eval.metrics.rtf,
                        '降噪后文件': eval.denoised_audio_path
                    }
                    all_data.append(row)
            
            df_detail = pd.DataFrame(all_data)
            df_detail.to_excel(writer, sheet_name='详细结果', index=False)
            
            # 2. 算法汇总表
            summary_data = []
            for algo_name, evaluations in results.items():
                if not evaluations:
                    continue
                
                # 计算各项指标的平均值和标准差
                metrics_summary = self._compute_summary(evaluations)
                
                row = {
                    '算法名称': algo_name,
                    '测试文件数': len(evaluations),
                    'PESQ均值': metrics_summary.get('pesq_mean'),
                    'PESQ标准差': metrics_summary.get('pesq_std'),
                    'STOI均值': metrics_summary.get('stoi_mean'),
                    'STOI标准差': metrics_summary.get('stoi_std'),
                    'SI-SDR均值(dB)': metrics_summary.get('sisdr_mean'),
                    'SI-SDR标准差': metrics_summary.get('sisdr_std'),
                    'DNSMOS OVRL均值': metrics_summary.get('dnsmos_ovrl_mean'),
                    'NISQA MOS均值': metrics_summary.get('nisqa_mos_mean'),
                    'UTMOS均值': metrics_summary.get('utmos_mean'),
                    '平均处理时间(s)': metrics_summary.get('processing_time_mean'),
                    '平均RTF': metrics_summary.get('rtf_mean')
                }
                summary_data.append(row)
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='算法汇总', index=False)
            
            # 3. 算法对比表
            if len(results) > 1:
                comparison_data = self._generate_comparison(results)
                df_comparison = pd.DataFrame(comparison_data)
                df_comparison.to_excel(writer, sheet_name='算法对比', index=False)
        
        print(f"Excel报告已生成: {output_file}")
        return output_file
    
    def generate_html_report(
        self,
        results: Dict[str, List[DenoiseEvaluation]],
        output_file: Optional[str] = None,
        title: str = "降噪算法测评报告"
    ) -> str:
        """
        生成HTML格式的可视化报告
        
        Args:
            results: 测评结果字典
            output_file: 输出文件路径(可选)
            title: 报告标题
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"denoise_report_{timestamp}.html")
        
        # 生成汇总数据
        summary_data = []
        for algo_name, evaluations in results.items():
            if not evaluations:
                continue
            metrics_summary = self._compute_summary(evaluations)
            summary_data.append({
                'algorithm': algo_name,
                'count': len(evaluations),
                **metrics_summary
            })
        
        # 构建HTML
        html_content = self._build_html(title, summary_data, results)
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {output_file}")
        return output_file
    
    def generate_markdown_report(
        self,
        results: Dict[str, List[DenoiseEvaluation]],
        output_file: Optional[str] = None,
        title: str = "降噪算法测评报告"
    ) -> str:
        """
        生成Markdown格式的报告
        
        Args:
            results: 测评结果字典
            output_file: 输出文件路径(可选)
            title: 报告标题
            
        Returns:
            输出文件路径
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"denoise_report_{timestamp}.md")
        
        lines = []
        lines.append(f"# {title}")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. 概述
        lines.append("## 1. 测评概述\n")
        lines.append(f"- **测试算法数**: {len(results)}")
        total_files = sum(len(evals) for evals in results.values())
        lines.append(f"- **总测试文件数**: {total_files}")
        lines.append("")
        
        # 2. 算法汇总
        lines.append("## 2. 算法性能汇总\n")
        lines.append("| 算法名称 | 文件数 | PESQ | STOI | SI-SDR(dB) | 处理时间(s) | RTF |")
        lines.append("|---------|--------|------|------|-----------|------------|-----|")
        
        for algo_name, evaluations in results.items():
            if not evaluations:
                continue
            
            metrics_summary = self._compute_summary(evaluations)
            
            pesq_str = f"{metrics_summary.get('pesq_mean', 0):.3f}±{metrics_summary.get('pesq_std', 0):.3f}" if metrics_summary.get('pesq_mean') else "N/A"
            stoi_str = f"{metrics_summary.get('stoi_mean', 0):.3f}±{metrics_summary.get('stoi_std', 0):.3f}" if metrics_summary.get('stoi_mean') else "N/A"
            sisdr_str = f"{metrics_summary.get('sisdr_mean', 0):.2f}±{metrics_summary.get('sisdr_std', 0):.2f}" if metrics_summary.get('sisdr_mean') else "N/A"
            time_str = f"{metrics_summary.get('processing_time_mean', 0):.3f}"
            rtf_str = f"{metrics_summary.get('rtf_mean', 0):.3f}"
            
            lines.append(f"| {algo_name} | {len(evaluations)} | {pesq_str} | {stoi_str} | {sisdr_str} | {time_str} | {rtf_str} |")
        
        lines.append("")
        
        # 3. 详细结果
        lines.append("## 3. 详细测评结果\n")
        
        for algo_name, evaluations in results.items():
            lines.append(f"### {algo_name}\n")
            
            if not evaluations:
                lines.append("*无测评数据*\n")
                continue
            
            lines.append("| 文件名 | PESQ | STOI | SI-SDR | DNSMOS | 处理时间 |")
            lines.append("|--------|------|------|--------|--------|----------|")
            
            for eval in evaluations[:10]:  # 只显示前10个
                pesq = f"{eval.metrics.pesq:.3f}" if eval.metrics.pesq else "N/A"
                stoi = f"{eval.metrics.stoi:.3f}" if eval.metrics.stoi else "N/A"
                sisdr = f"{eval.metrics.sisdr:.2f}" if eval.metrics.sisdr else "N/A"
                dnsmos = f"{eval.metrics.dnsmos_ovrl:.3f}" if eval.metrics.dnsmos_ovrl else "N/A"
                time_str = f"{eval.metrics.processing_time:.3f}"
                
                lines.append(f"| {eval.file_name} | {pesq} | {stoi} | {sisdr} | {dnsmos} | {time_str} |")
            
            if len(evaluations) > 10:
                lines.append(f"| ... ({len(evaluations) - 10} more) | | | | | |")
            
            lines.append("")
        
        # 4. 结论和建议
        lines.append("## 4. 结论与建议\n")
        lines.append(self._generate_conclusions(results))
        lines.append("")
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Markdown报告已生成: {output_file}")
        return output_file
    
    def _compute_summary(self, evaluations: List[DenoiseEvaluation]) -> dict:
        """计算测评数据的汇总统计"""
        summary = {}
        
        # 收集各项指标
        metrics_dict = {
            'pesq': [],
            'stoi': [],
            'sisdr': [],
            'dnsmos_ovrl': [],
            'dnsmos_sig': [],
            'dnsmos_bak': [],
            'nisqa_mos': [],
            'utmos': [],
            'processing_time': [],
            'rtf': []
        }
        
        for eval in evaluations:
            for key in metrics_dict.keys():
                value = getattr(eval.metrics, key)
                if value is not None:
                    metrics_dict[key].append(value)
        
        # 计算均值和标准差
        for key, values in metrics_dict.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
            else:
                summary[f'{key}_mean'] = None
                summary[f'{key}_std'] = None
        
        return summary
    
    def _generate_comparison(self, results: Dict[str, List[DenoiseEvaluation]]) -> List[dict]:
        """生成算法对比数据"""
        comparison = []
        
        # 找出最佳算法
        best_algorithms = {}
        
        for metric in ['pesq', 'stoi', 'sisdr']:
            best_value = -np.inf
            best_algo = None
            
            for algo_name, evaluations in results.items():
                if not evaluations:
                    continue
                
                values = [getattr(e.metrics, metric) for e in evaluations if getattr(e.metrics, metric) is not None]
                if values:
                    mean_value = np.mean(values)
                    if mean_value > best_value:
                        best_value = mean_value
                        best_algo = algo_name
            
            if best_algo:
                best_algorithms[metric] = best_algo
        
        # 生成对比表
        for algo_name, evaluations in results.items():
            if not evaluations:
                continue
            
            summary = self._compute_summary(evaluations)
            
            row = {
                '算法名称': algo_name,
                'PESQ排名': '最佳' if best_algorithms.get('pesq') == algo_name else '',
                'STOI排名': '最佳' if best_algorithms.get('stoi') == algo_name else '',
                'SI-SDR排名': '最佳' if best_algorithms.get('sisdr') == algo_name else '',
                '综合评分': self._compute_overall_score(summary)
            }
            comparison.append(row)
        
        # 按综合评分排序
        comparison.sort(key=lambda x: x['综合评分'], reverse=True)
        
        return comparison
    
    def _compute_overall_score(self, summary: dict) -> float:
        """计算综合评分"""
        scores = []
        
        # PESQ (权重 0.3)
        if summary.get('pesq_mean'):
            scores.append(summary['pesq_mean'] * 0.3 / 4.5)  # 归一化到0-1
        
        # STOI (权重 0.3)
        if summary.get('stoi_mean'):
            scores.append(summary['stoi_mean'] * 0.3)
        
        # SI-SDR (权重 0.2)
        if summary.get('sisdr_mean'):
            sisdr_score = min(max(summary['sisdr_mean'] / 20, 0), 1)  # 假设20dB为满分
            scores.append(sisdr_score * 0.2)
        
        # DNSMOS (权重 0.2)
        if summary.get('dnsmos_ovrl_mean'):
            scores.append(summary['dnsmos_ovrl_mean'] * 0.2)
        
        return sum(scores) if scores else 0.0
    
    def _generate_conclusions(self, results: Dict[str, List[DenoiseEvaluation]]) -> str:
        """生成结论和建议"""
        conclusions = []
        
        # 找出各项指标最佳的算法
        best_pesq = None
        best_stoi = None
        best_sisdr = None
        best_rtf = None
        
        for algo_name, evaluations in results.items():
            if not evaluations:
                continue
            
            summary = self._compute_summary(evaluations)
            
            if summary.get('pesq_mean'):
                if best_pesq is None or summary['pesq_mean'] > best_pesq[1]:
                    best_pesq = (algo_name, summary['pesq_mean'])
            
            if summary.get('stoi_mean'):
                if best_stoi is None or summary['stoi_mean'] > best_stoi[1]:
                    best_stoi = (algo_name, summary['stoi_mean'])
            
            if summary.get('sisdr_mean'):
                if best_sisdr is None or summary['sisdr_mean'] > best_sisdr[1]:
                    best_sisdr = (algo_name, summary['sisdr_mean'])
            
            if summary.get('rtf_mean'):
                if best_rtf is None or summary['rtf_mean'] < best_rtf[1]:
                    best_rtf = (algo_name, summary['rtf_mean'])
        
        # 生成结论
        if best_pesq:
            conclusions.append(f"- **PESQ最佳**: {best_pesq[0]} (得分: {best_pesq[1]:.3f})")
        if best_stoi:
            conclusions.append(f"- **STOI最佳**: {best_stoi[0]} (得分: {best_stoi[1]:.3f})")
        if best_sisdr:
            conclusions.append(f"- **SI-SDR最佳**: {best_sisdr[0]} (得分: {best_sisdr[1]:.2f} dB)")
        if best_rtf:
            conclusions.append(f"- **实时性最佳**: {best_rtf[0]} (RTF: {best_rtf[1]:.3f})")
        
        if not conclusions:
            conclusions.append("*暂无足够数据生成结论*")
        
        return '\n'.join(conclusions)
    
    def _build_html(self, title: str, summary_data: List[dict], results: Dict[str, List[DenoiseEvaluation]]) -> str:
        """构建HTML报告内容"""
        # 简化的HTML模板
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 5px; padding: 8px 15px; background: #e3f2fd; border-radius: 4px; }}
        .best {{ background: #c8e6c9; font-weight: bold; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 算法性能汇总</h2>
        <table>
            <tr>
                <th>算法名称</th>
                <th>文件数</th>
                <th>PESQ</th>
                <th>STOI</th>
                <th>SI-SDR (dB)</th>
                <th>处理时间(s)</th>
                <th>RTF</th>
            </tr>
"""
        
        for data in summary_data:
            algo = data['algorithm']
            count = data['count']
            pesq = f"{data.get('pesq_mean', 0):.3f}" if data.get('pesq_mean') else "N/A"
            stoi = f"{data.get('stoi_mean', 0):.3f}" if data.get('stoi_mean') else "N/A"
            sisdr = f"{data.get('sisdr_mean', 0):.2f}" if data.get('sisdr_mean') else "N/A"
            time_str = f"{data.get('processing_time_mean', 0):.3f}"
            rtf = f"{data.get('rtf_mean', 0):.3f}" if data.get('rtf_mean') else "N/A"
            
            html += f"""
            <tr>
                <td><strong>{algo}</strong></td>
                <td>{count}</td>
                <td>{pesq}</td>
                <td>{stoi}</td>
                <td>{sisdr}</td>
                <td>{time_str}</td>
                <td>{rtf}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>📈 指标说明</h2>
        <div>
            <span class="metric">PESQ: 感知语音质量 (1-4.5, 越高越好)</span>
            <span class="metric">STOI: 短时客观可懂度 (0-1, 越高越好)</span>
            <span class="metric">SI-SDR: 尺度不变信噪比 (dB, 越高越好)</span>
            <span class="metric">RTF: 实时因子 (&lt;1表示实时)</span>
        </div>
        
        <div class="footer">
            <p>AudioMOS 降噪算法测评系统</p>
        </div>
    </div>
</body>
</html>
"""
        return html
