"""
ASR Benchmark评测引擎
支持多算法横向对比评测，在标准数据集上计算CER/WER/RTF等指标
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
from .evaluator import ASRMetrics, compute_cer, evaluate_asr
from .dataset_manager import DatasetSample

logger = logging.getLogger("audiomos")


@dataclass
class BenchmarkResult:
    """单算法Benchmark结果"""
    algorithm_name: str
    metrics: ASRMetrics = field(default_factory=ASRMetrics)
    per_utterance: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "algorithm_name": self.algorithm_name,
            "metrics": self.metrics.to_dict(),
            "per_utterance": self.per_utterance[:20],  # 只返回前20条详情
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

    def __init__(self, model_dir: str = "./models/asr", device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self._runs: Dict[str, BenchmarkRun] = {}

    def run_benchmark(
        self,
        algorithms: List[str],
        samples: List[DatasetSample],
        dataset_name: str = "custom",
        bench_id: Optional[str] = None,
    ) -> BenchmarkRun:
        """
        运行Benchmark评测

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
        completed = 0

        logger.info(f"[Benchmark] 开始评测: bench_id={bench_id}, 算法数={len(algorithms)}, 数据集={dataset_name}, 样本数={len(samples)}, 总任务数={total_tasks}")

        for algo_name in algorithms:
            algo_result = BenchmarkResult(algorithm_name=algo_name)
            algo_task_count = len(samples)
            logger.info(f"[Benchmark] 开始评测算法: {algo_name} ({algo_task_count} 条样本)")

            # 获取/初始化算法
            instance = ASRRegistry.get(algo_name, device=self.device, model_dir=os.path.join(self.model_dir, algo_name))
            if not instance:
                algo_result.errors.append(f"算法 {algo_name} 不可用")
                run.results[algo_name] = algo_result
                completed += len(samples)
                run.progress = completed / total_tasks * 100
                continue

            if not instance.is_initialized():
                try:
                    instance.initialize()
                except Exception as e:
                    algo_result.errors.append(f"初始化失败: {e}")
                    run.results[algo_name] = algo_result
                    completed += len(samples)
                    run.progress = completed / total_tasks * 100
                    continue

            # 逐条评测
            references = []
            hypotheses = []
            proc_times = []
            durations = []

            for idx, sample in enumerate(samples):
                try:
                    import soundfile as sf
                    audio, sr = sf.read(sample.audio_path)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    audio_dur = len(audio) / sr

                    # 直接调用 transcribe，避免 transcribe_file 重复读取文件
                    start_time = time.time()
                    asr_result = instance.transcribe(audio, sr)
                    asr_result.processing_time = time.time() - start_time
                    asr_result.rtf = asr_result.processing_time / audio_dur if audio_dur > 0 else 0
                    asr_result.algorithm_name = instance.name

                    references.append(sample.reference_text)
                    hypotheses.append(asr_result.text)
                    proc_times.append(asr_result.processing_time)
                    durations.append(audio_dur)

                    cer, _, _, _ = compute_cer(sample.reference_text, asr_result.text)
                    algo_result.per_utterance.append({
                        "utterance_id": sample.utterance_id,
                        "reference": sample.reference_text[:100],
                        "hypothesis": asr_result.text[:100],
                        "cer": round(cer, 4),
                        "processing_time": round(asr_result.processing_time, 3),
                    })

                    # 逐条打印识别结果
                    logger.info(f"[Benchmark] {algo_name} [{idx+1}/{algo_task_count}] "
                                f"utt={sample.utterance_id} CER={cer:.4f} "
                                f"ref={sample.reference_text[:60]} "
                                f"hyp={asr_result.text[:60]}")

                except Exception as e:
                    algo_result.errors.append(f"{sample.utterance_id}: {e}")
                    logger.warning(f"[Benchmark] {algo_name} [{idx+1}/{algo_task_count}] "
                                   f"utt={sample.utterance_id} 错误: {e}")

                completed += 1
                run.progress = completed / total_tasks * 100

            # 计算汇总指标
            if references:
                algo_result.metrics = evaluate_asr(references, hypotheses, proc_times, durations)
                logger.info(f"[Benchmark] {algo_name} 完成: CER={algo_result.metrics.cer:.4f} "
                            f"WER={algo_result.metrics.wer:.4f} RTF={algo_result.metrics.rtf:.4f}")

            run.results[algo_name] = algo_result

        run.status = "completed"
        run.end_time = datetime.now().isoformat()
        run.progress = 100.0

        # 汇总日志
        elapsed = (datetime.fromisoformat(run.end_time) - datetime.fromisoformat(run.start_time)).total_seconds()
        algo_summaries = []
        for name, r in run.results.items():
            m = r.metrics
            algo_summaries.append(f"{name}: CER={m.cer:.4f} WER={m.wer:.4f} RTF={m.rtf:.4f}" if m.cer is not None else f"{name}: 失败")
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
