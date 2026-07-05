"""
ASR评测报告生成模块
支持JSON/Excel/HTML/Markdown多格式报告
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from .benchmark import BenchmarkRun, BenchmarkResult

logger = logging.getLogger("audiomos")


class ASRReportGenerator:
    """ASR评测报告生成器"""

    @staticmethod
    def generate_json(run: BenchmarkRun, output_path: str) -> str:
        """生成JSON报告"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, ensure_ascii=False, indent=2)
        return output_path

    @staticmethod
    def generate_markdown(run: BenchmarkRun, output_path: str) -> str:
        """生成Markdown报告"""
        lines = []
        lines.append(f"# ASR Benchmark 评测报告")
        lines.append(f"")
        lines.append(f"- **评测ID**: {run.bench_id}")
        lines.append(f"- **数据集**: {run.dataset_name}")
        lines.append(f"- **评测算法**: {', '.join(run.algorithms)}")
        lines.append(f"- **开始时间**: {run.start_time}")
        lines.append(f"- **结束时间**: {run.end_time}")
        lines.append(f"- **状态**: {run.status}")
        lines.append(f"")

        # 排名表
        lines.append(f"## 排名")
        lines.append(f"")
        lines.append(f"| 排名 | 算法 | CER | WER | RTF | 评测句数 |")
        lines.append(f"|------|------|-----|-----|-----|----------|")

        ranking = []
        for name, result in run.results.items():
            ranking.append({
                "algorithm": name,
                "cer": result.metrics.cer,
                "wer": result.metrics.wer,
                "rtf": result.metrics.rtf,
                "num_utterances": result.metrics.num_utterances,
            })
        ranking.sort(key=lambda x: x.get("cer", float("inf")))

        for i, r in enumerate(ranking, 1):
            lines.append(
                f"| {i} | {r['algorithm']} | "
                f"{r['cer']:.4f} | {r['wer']:.4f} | "
                f"{r['rtf']:.4f} | {r['num_utterances']} |"
            )

        lines.append(f"")

        # 详细指标
        lines.append(f"## 详细指标")
        lines.append(f"")
        for name, result in run.results.items():
            m = result.metrics
            lines.append(f"### {name}")
            lines.append(f"")
            lines.append(f"| 指标 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| CER | {m.cer:.4f} |")
            lines.append(f"| CER-删除 | {m.cer_del:.4f} |")
            lines.append(f"| CER-插入 | {m.cer_ins:.4f} |")
            lines.append(f"| CER-替换 | {m.cer_sub:.4f} |")
            lines.append(f"| WER | {m.wer:.4f} |")
            lines.append(f"| RTF | {m.rtf:.4f} |")
            lines.append(f"| 总处理时间 | {m.processing_time:.2f}s |")
            lines.append(f"| 音频总时长 | {m.audio_duration:.2f}s |")
            lines.append(f"| 评测句数 | {m.num_utterances} |")
            lines.append(f"")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return output_path

    @staticmethod
    def generate_excel(run: BenchmarkRun, output_path: str) -> str:
        """生成Excel报告"""
        try:
            import pandas as pd

            # 汇总表
            summary_rows = []
            for name, result in run.results.items():
                m = result.metrics
                summary_rows.append({
                    "算法": name,
                    "CER": m.cer,
                    "CER_删除": m.cer_del,
                    "CER_插入": m.cer_ins,
                    "CER_替换": m.cer_sub,
                    "WER": m.wer,
                    "RTF": m.rtf,
                    "总处理时间(s)": m.processing_time,
                    "音频总时长(s)": m.audio_duration,
                    "评测句数": m.num_utterances,
                })

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df_summary = pd.DataFrame(summary_rows)
                df_summary = df_summary.sort_values("CER")
                df_summary.to_excel(writer, sheet_name="汇总排名", index=False)

                # 逐条详情
                for name, result in run.results.items():
                    if result.per_utterance:
                        df_detail = pd.DataFrame(result.per_utterance)
                        sheet_name = name[:31]  # Excel sheet名最长31字符
                        df_detail.to_excel(writer, sheet_name=sheet_name, index=False)

        except ImportError:
            logger.warning("pandas/openpyxl不可用，回退到JSON报告")
            return ASRReportGenerator.generate_json(run, output_path.replace(".xlsx", ".json"))

        return output_path

    @staticmethod
    def generate_html(run: BenchmarkRun, output_path: str) -> str:
        """生成HTML报告"""
        ranking = []
        for name, result in run.results.items():
            ranking.append({
                "algorithm": name,
                "cer": result.metrics.cer,
                "wer": result.metrics.wer,
                "rtf": result.metrics.rtf,
                "num_utterances": result.metrics.num_utterances,
            })
        ranking.sort(key=lambda x: x.get("cer", float("inf")))

        # 构建HTML
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ASR Benchmark 评测报告 - {run.bench_id}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #1a1a2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background-color: #1a1a2e; color: white; }}
tr:nth-child(even) {{ background-color: #f8f9fa; }}
.rank-1 {{ background-color: #ffd700 !important; font-weight: bold; }}
.rank-2 {{ background-color: #c0c0c0 !important; }}
.rank-3 {{ background-color: #cd7f32 !important; }}
</style>
</head>
<body>
<h1>ASR Benchmark 评测报告</h1>
<p>评测ID: {run.bench_id} | 数据集: {run.dataset_name} | 时间: {run.start_time}</p>

<h2>排名</h2>
<table>
<tr><th>排名</th><th>算法</th><th>CER</th><th>WER</th><th>RTF</th><th>评测句数</th></tr>
"""

        for i, r in enumerate(ranking, 1):
            rank_class = f"rank-{i}" if i <= 3 else ""
            html += f'<tr class="{rank_class}"><td>{i}</td><td>{r["algorithm"]}</td>'
            html += f'<td>{r["cer"]:.4f}</td><td>{r["wer"]:.4f}</td>'
            html += f'<td>{r["rtf"]:.4f}</td><td>{r["num_utterances"]}</td></tr>\n'

        html += """</table>
<h2>详细指标</h2>
"""

        for name, result in run.results.items():
            m = result.metrics
            html += f"""<h3>{name}</h3>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>CER</td><td>{m.cer:.4f}</td></tr>
<tr><td>CER-删除</td><td>{m.cer_del:.4f}</td></tr>
<tr><td>CER-插入</td><td>{m.cer_ins:.4f}</td></tr>
<tr><td>CER-替换</td><td>{m.cer_sub:.4f}</td></tr>
<tr><td>WER</td><td>{m.wer:.4f}</td></tr>
<tr><td>RTF</td><td>{m.rtf:.4f}</td></tr>
<tr><td>总处理时间</td><td>{m.processing_time:.2f}s</td></tr>
<tr><td>音频总时长</td><td>{m.audio_duration:.2f}s</td></tr>
</table>
"""

        html += "</body></html>"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    @staticmethod
    def generate(run: BenchmarkRun, output_dir: str, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        生成多格式报告

        Args:
            run: Benchmark运行结果
            output_dir: 输出目录
            formats: 格式列表 ["json", "markdown", "excel", "html"]

        Returns:
            {格式: 文件路径}
        """
        os.makedirs(output_dir, exist_ok=True)
        formats = formats or ["json", "markdown", "html"]
        results = {}

        for fmt in formats:
            try:
                if fmt == "json":
                    path = os.path.join(output_dir, f"{run.bench_id}.json")
                    results[fmt] = ASRReportGenerator.generate_json(run, path)
                elif fmt == "markdown":
                    path = os.path.join(output_dir, f"{run.bench_id}.md")
                    results[fmt] = ASRReportGenerator.generate_markdown(run, path)
                elif fmt == "excel":
                    path = os.path.join(output_dir, f"{run.bench_id}.xlsx")
                    results[fmt] = ASRReportGenerator.generate_excel(run, path)
                elif fmt == "html":
                    path = os.path.join(output_dir, f"{run.bench_id}.html")
                    results[fmt] = ASRReportGenerator.generate_html(run, path)
            except Exception as e:
                logger.error(f"生成{fmt}报告失败: {e}")

        return results
