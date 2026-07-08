"""
ASR Benchmark评测引擎
支持多算法横向对比评测，在标准数据集上计算CER/WER/RTF等指标
支持并行评测多算法（可配置并发数）
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseASR, ASRResult
from .registry import ASRRegistry
from .evaluator import ASRMetrics, evaluate_asr
from .dataset_manager import DatasetSample

logger = logging.getLogger("audiomos")

# 结果保留截断数（可从配置覆盖）
DEFAULT_PER_UTTERANCE_LIMIT = 20


@dataclass
class BenchmarkResult:
    """单算法Benchmark结果"""
    algorithm_name: str
    metrics: ASRMetrics = field(default_factory=ASRMetrics)
    per_utterance: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self, per_utterance_limit: int = DEFAULT_PER_UTTERANCE_LIMIT) -> dict:
        return {
            "algorithm_name": self.algorithm_name,
            "metrics": self.metrics.to_dict(),
            "per_utterance": self.per_utterance[:per_utterance_limit],
            "errors": self.errors,
        }


@dataclass
class BenchmarkRun:
    """一次Benchmark运行"""
    bench_id: str
    algorithms: List[str]
    dataset_name: str
    status: str = "pending"  # pending/running/completed/failed
    progress: float = 0.0
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bench_id": self.bench_id,
            "algorithms": self.algorithms,
            "dataset_name": self.dataset_name,
            "status": self.status,
            "progress": round(self.progress, 2),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
        }


class ASRBenchmark:
    """ASR Benchmark评测引擎"""

    def __init__(self, model_dir: str = "./models/asr", device: str = "cuda",
                 max_workers: int = 1, per_utterance_limit: int = DEFAULT_PER_UTTERANCE_LIMIT):
        self.model_dir = model_dir
        self.device = device
        self.max_workers = max_workers
        self.per_utterance_limit = per_utterance_limit
        self._runs: Dict[str, BenchmarkRun] = {}

    def _run_single_algorithm(
        self,
        algo_name: str,
        samples: List[DatasetSample],
    ) -> BenchmarkResult:
        """在单个线程中评测一个算法（供 ThreadPoolExecutor 调用）"""
        algo_result = BenchmarkResult(algorithm_name=algo_name)
        algo_task_count = len(samples)
        logger.info(f"[Benchmark] 开始评测算法: {algo_name} ({algo_task_count} 条样本)")

        # 获取/初始化算法
        instance = ASRRegistry.get(
            algo_name, device=self.device,
            model_dir=os.path.join(self.model_dir, algo_name)
        )
        if not instance:
            algo_result.errors.append(f"算法 {algo_name} 不可用")
            return algo_result

        if not instance.is_initialized():
            try:
                instance.initialize()
            except Exception as e:
                algo_result.errors.append(f"初始化失败: {e}")
                return algo_result

        # 逐条评测 — CER/WER 在 evaluate_asr 中统一计算，此处仅收集结果
        references = []
        hypotheses = []
        proc_times = []
        durations = []
        raw_results = []  # 暂存每条样本的信息，等 evaluate_asr 完成后补填 CER/WER

        for idx, sample in enumerate(samples):
            try:
                import soundfile as sf
                audio, sr = sf.read(sample.audio_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audio_dur = len(audio) / sr

                start_time = time.time()
                asr_result = instance.transcribe(audio, sr)
                asr_result.processing_time = time.time() - start_time
                asr_result.rtf = asr_result.processing_time / audio_dur if audio_dur > 0 else 0
                asr_result.algorithm_name = instance.name

                references.append(sample.reference_text)
                hypotheses.append(asr_result.text)
                proc_times.append(asr_result.processing_time)
                durations.append(audio_dur)

                raw_results.append({
                    "utterance_id": sample.utterance_id,
                    "reference": sample.reference_text[:100],
                    "hypothesis": asr_result.text[:100],
                    "processing_time": round(asr_result.processing_time, 3),
                })

                logger.info(f"[Benchmark] {algo_name} [{idx+1}/{algo_task_count}] "
                            f"utt={sample.utterance_id} "
                            f"ref={sample.reference_text[:60]} "
                            f"hyp={asr_result.text[:60]}")

            except Exception as e:
                algo_result.errors.append(f"{sample.utterance_id}: {e}")
                logger.warning(f"[Benchmark] {algo_name} [{idx+1}/{algo_task_count}] "
                               f"utt={sample.utterance_id} 错误: {e}")

        # 统一计算汇总指标（一次 evaluate_asr，同时产出 CER/WER + per_utterance）
        if references:
            algo_result.metrics = evaluate_asr(references, hypotheses, proc_times, durations)

            # 从 metrics.per_utterance_cer/wer 回填到每条详情
            for i, raw in enumerate(raw_results):
                if i < len(algo_result.metrics.per_utterance_cer):
                    raw["cer"] = round(algo_result.metrics.per_utterance_cer[i], 4)
                if i < len(algo_result.metrics.per_utterance_wer):
                    raw["wer"] = round(algo_result.metrics.per_utterance_wer[i], 4)
            algo_result.per_utterance = raw_results

            logger.info(f"[Benchmark] {algo_name} 完成: CER={algo_result.metrics.cer:.4f} "
                        f"WER={algo_result.metrics.wer:.4f} RTF={algo_result.metrics.rtf:.4f}")

        return algo_result

    def run_benchmark(
        self,
        algorithms: List[str],
        samples: List[DatasetSample],
        dataset_name: str = "custom",
        bench_id: Optional[str] = None,
    ) -> BenchmarkRun:
        """
        运行Benchmark评测（支持并行多算法）

        Args:
            algorithms: 算法名称列表
            samples: 测试样本列表
            dataset_name: 数据集名称
            bench_id: 运行ID

        Returns:
            BenchmarkRun
        """
        import uuid
        bench_id = bench_id or f"bench_{uuid.uuid4().hex[:8]}"

        run = BenchmarkRun(
            bench_id=bench_id,
            algorithms=algorithms,
            dataset_name=dataset_name,
            status="running",
            start_time=datetime.now().isoformat(),
        )
        self._runs[bench_id] = run

        total_tasks = len(algorithms) * len(samples)
        logger.info(f"[Benchmark] 开始评测: bench_id={bench_id}, 算法数={len(algorithms)}, "
                    f"数据集={dataset_name}, 样本数={len(samples)}, "
                    f"并发数={self.max_workers}, 总任务数={total_tasks}")

        completed = 0
        algo_order = []  # 保持原始顺序

        if self.max_workers <= 1 or len(algorithms) <= 1:
            # 串行模式（默认）
            for algo_name in algorithms:
                algo_result = self._run_single_algorithm(algo_name, samples)
                run.results[algo_name] = algo_result
                completed += len(samples)
                run.progress = completed / total_tasks * 100
        else:
            # 并行模式 — 每个算法一个线程
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._run_single_algorithm, algo, samples): algo
                    for algo in algorithms
                }
                algo_order = algorithms  # 记录顺序
                for future in as_completed(futures):
                    algo_name = futures[future]
                    try:
                        algo_result = future.result()
                    except Exception as e:
                        logger.error(f"[Benchmark] 算法 {algo_name} 执行异常: {e}")
                        algo_result = BenchmarkResult(algorithm_name=algo_name)
                        algo_result.errors.append(str(e))
                    run.results[algo_name] = algo_result
                    completed += len(samples)
                    run.progress = completed / total_tasks * 100

        run.status = "completed"
        run.end_time = datetime.now().isoformat()
        run.progress = 100.0

        # 汇总日志
        elapsed = (datetime.fromisoformat(run.end_time) - datetime.fromisoformat(run.start_time)).total_seconds()
        algo_summaries = []
        for name in algorithms:
            r = run.results.get(name)
            if r and r.metrics.cer is not None:
                m = r.metrics
                algo_summaries.append(f"{name}: CER={m.cer:.4f} WER={m.wer:.4f} RTF={m.rtf:.4f}")
            else:
                algo_summaries.append(f"{name}: 失败")
        logger.info(f"[Benchmark] 评测完成: bench_id={bench_id} 耗时={elapsed:.1f}s | " + " | ".join(algo_summaries))

        return run

    def get_run(self, bench_id: str) -> Optional[BenchmarkRun]:
        """获取Benchmark运行结果"""
        return self._runs.get(bench_id)

    def list_runs(self) -> List[dict]:
        """列出所有Benchmark运行"""
        return [r.to_dict() for r in self._runs.values()]

    def get_ranking(self, bench_id: str) -> List[dict]:
        """获取排名（按CER升序）"""
        run = self._runs.get(bench_id)
        if not run:
            return []

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
        return ranking
