"""
降噪算法计算效率分析模块

系统性地评估降噪算法的计算资源消耗。
- RTF (Real-Time Factor): 处理时间/音频时长
- CPU/GPU内存使用
- GPU利用率
- 模型参数量和文件大小
- 吞吐量 (批量处理)

使用方式:
    profiler = EfficiencyProfiler()
    metrics = profiler.profile_denoiser(denoiser, "path/to/audio.wav")
    print(f"RTF: {metrics.rtf:.3f}, 峰值内存: {metrics.peak_memory_mb:.1f}MB")
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ===========================
# 硬件监控（可选依赖）
# ===========================

_PSUTIL_AVAILABLE = False
_PYNVML_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass

try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
    _NVML_DEVICE_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    _PYNVML_AVAILABLE = False
    _NVML_DEVICE_COUNT = 0


# ===========================
# 数据类
# ===========================


@dataclass
class EfficiencyMetrics:
    """效率分析指标"""
    # 时间指标
    processing_time: float = 0.0     # 总处理时间 (秒)
    audio_duration: float = 0.0      # 音频时长 (秒)
    rtf: float = 0.0                 # 实时因子 (处理时间/音频时长, <1表示实时)

    # 内存指标
    peak_memory_mb: float = 0.0      # 峰值CPU内存 (MB)
    avg_memory_mb: float = 0.0       # 平均CPU内存 (MB)
    peak_gpu_memory_mb: float = 0.0  # 峰值GPU显存 (MB)
    avg_gpu_memory_mb: float = 0.0   # 平均GPU显存 (MB)

    # GPU指标
    gpu_utilization_pct: float = 0.0 # GPU利用率 (%)

    # 模型指标
    model_params_count: int = 0      # 模型参数量
    model_size_mb: float = 0.0       # 模型文件大小 (MB)

    # 音频指标
    input_sample_rate: int = 0       # 输入采样率
    input_channels: int = 1          # 输入通道数

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "processing_time": round(self.processing_time, 4),
            "audio_duration": round(self.audio_duration, 2),
            "rtf": round(self.rtf, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "avg_memory_mb": round(self.avg_memory_mb, 1),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 1),
            "avg_gpu_memory_mb": round(self.avg_gpu_memory_mb, 1),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "model_params_count": self.model_params_count,
            "model_size_mb": round(self.model_size_mb, 2),
            "input_sample_rate": self.input_sample_rate,
            "input_channels": self.input_channels,
        }


@dataclass
class EfficiencyReport:
    """批量效率分析报告"""
    algorithm_name: str
    metrics_list: List[EfficiencyMetrics] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)  # 汇总统计

    @property
    def avg_rtf(self) -> float:
        if not self.metrics_list:
            return 0.0
        return float(np.mean([m.rtf for m in self.metrics_list]))

    @property
    def is_realtime(self) -> bool:
        return self.avg_rtf < 1.0


# ===========================
# 硬件监控
# ===========================


class _HardwareMonitor:
    """硬件资源监控"""

    def __init__(self, interval: float = 0.1):
        """
        Args:
            interval: 采样间隔 (秒)
        """
        self.interval = interval
        self._running = False
        self._thread = None
        self._samples_cpu_mb: List[float] = []
        self._samples_gpu_mb: List[float] = []
        self._samples_gpu_util: List[float] = []

    def start(self):
        """开始监控"""
        self._running = True
        self._samples_cpu_mb = []
        self._samples_gpu_mb = []
        self._samples_gpu_util = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict:
        """停止监控并返回统计"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        return {
            "peak_memory_mb": max(self._samples_cpu_mb) if self._samples_cpu_mb else 0.0,
            "avg_memory_mb": np.mean(self._samples_cpu_mb) if self._samples_cpu_mb else 0.0,
            "peak_gpu_memory_mb": max(self._samples_gpu_mb) if self._samples_gpu_mb else 0.0,
            "avg_gpu_memory_mb": np.mean(self._samples_gpu_mb) if self._samples_gpu_mb else 0.0,
            "gpu_utilization_pct": np.mean(self._samples_gpu_util) if self._samples_gpu_util else 0.0,
            "n_samples": len(self._samples_cpu_mb),
        }

    def _monitor_loop(self):
        """监控循环（在后台线程运行）"""
        while self._running:
            try:
                # CPU内存
                if _PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    mem_info = process.memory_info()
                    self._samples_cpu_mb.append(mem_info.rss / (1024 * 1024))

                # GPU内存和利用率
                if _PYNVML_AVAILABLE and _NVML_DEVICE_COUNT > 0:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self._samples_gpu_mb.append(mem.used / (1024 * 1024))
                        self._samples_gpu_util.append(util.gpu)
                    except Exception:
                        pass

            except Exception:
                pass

            time.sleep(self.interval)


# ===========================
# 效率分析器
# ===========================


class EfficiencyProfiler:
    """
    降噪算法计算效率分析器

    使用方式:
        profiler = EfficiencyProfiler()
        metrics = profiler.profile_denoiser(denoiser, "path/to/audio.wav")
        print(f"RTF: {metrics.rtf:.3f}")
    """

    def __init__(self, monitor_interval: float = 0.05):
        """
        初始化分析器

        Args:
            monitor_interval: 硬件监控采样间隔 (秒)
        """
        self.monitor_interval = monitor_interval

    def profile_denoiser(
        self,
        denoiser,  # BaseDenoiser 实例
        audio_file: str,
        warmup: bool = True,
    ) -> EfficiencyMetrics:
        """
        对单个降噪算法进行效率分析

        Args:
            denoiser: 降噪算法实例
            audio_file: 音频文件路径
            warmup: 是否先预热运行一次

        Returns:
            EfficiencyMetrics
        """
        metrics = EfficiencyMetrics()

        # 模型信息
        metrics.model_params_count = self._count_parameters(denoiser)
        metrics.model_size_mb = self._estimate_model_size(denoiser)

        # 音频信息
        try:
            info = sf.info(audio_file)
            metrics.audio_duration = info.duration
            metrics.input_sample_rate = info.samplerate
            metrics.input_channels = info.channels
        except Exception:
            pass

        # 预热（消除首轮加载开销）
        if warmup:
            try:
                audio, sr = sf.read(audio_file)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                denoiser.denoise(audio, sr)
            except Exception:
                pass

        # 加载音频
        audio, sr = sf.read(audio_file)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 开始监控
        monitor = _HardwareMonitor(interval=self.monitor_interval)
        monitor.start()

        # 计时推理
        start_time = time.perf_counter()
        try:
            result = denoiser.denoise(audio, sr)
        except Exception as e:
            logger.error(f"降噪失败: {e}")
            monitor.stop()
            return metrics
        end_time = time.perf_counter()

        # 停止监控
        hw_stats = monitor.stop()

        # 填充指标
        metrics.processing_time = end_time - start_time
        if metrics.audio_duration > 0:
            metrics.rtf = metrics.processing_time / metrics.audio_duration

        metrics.peak_memory_mb = hw_stats.get("peak_memory_mb", 0.0)
        metrics.avg_memory_mb = hw_stats.get("avg_memory_mb", 0.0)
        metrics.peak_gpu_memory_mb = hw_stats.get("peak_gpu_memory_mb", 0.0)
        metrics.avg_gpu_memory_mb = hw_stats.get("avg_gpu_memory_mb", 0.0)
        metrics.gpu_utilization_pct = hw_stats.get("gpu_utilization_pct", 0.0)

        return metrics

    def profile_batch(
        self,
        denoiser,
        audio_files: List[str],
        warmup: bool = True,
    ) -> EfficiencyReport:
        """
        对批量音频文件进行效率分析

        Args:
            denoiser: 降噪算法实例
            audio_files: 音频文件列表
            warmup: 是否预热

        Returns:
            EfficiencyReport
        """
        report = EfficiencyReport(
            algorithm_name=denoiser.name if hasattr(denoiser, 'name') else "Unknown",
        )

        for i, audio_file in enumerate(audio_files):
            logger.info(f"效率分析 [{i+1}/{len(audio_files)}]: {os.path.basename(audio_file)}")
            try:
                do_warmup = warmup and (i == 0)
                metrics = self.profile_denoiser(denoiser, audio_file, warmup=do_warmup)
                report.metrics_list.append(metrics)
            except Exception as e:
                logger.error(f"效率分析失败 {audio_file}: {e}")

        # 汇总统计
        if report.metrics_list:
            rtf_values = [m.rtf for m in report.metrics_list]
            times = [m.processing_time for m in report.metrics_list]
            memories = [m.peak_memory_mb for m in report.metrics_list if m.peak_memory_mb > 0]
            gpu_mems = [m.peak_gpu_memory_mb for m in report.metrics_list if m.peak_gpu_memory_mb > 0]

            report.summary = {
                "n_files": len(report.metrics_list),
                "rtf_mean": round(float(np.mean(rtf_values)), 4),
                "rtf_std": round(float(np.std(rtf_values)), 4),
                "rtf_min": round(float(np.min(rtf_values)), 4),
                "rtf_max": round(float(np.max(rtf_values)), 4),
                "is_realtime": float(np.mean(rtf_values)) < 1.0,
                "avg_processing_time": round(float(np.mean(times)), 4),
                "throughput_audio_per_sec": round(1.0 / float(np.mean(rtf_values)), 2) if float(np.mean(rtf_values)) > 0 else 0,
                "peak_memory_mb_mean": round(float(np.mean(memories)), 1) if memories else 0,
                "peak_gpu_memory_mb_mean": round(float(np.mean(gpu_mems)), 1) if gpu_mems else 0,
            }

        return report

    def profile_algorithms(
        self,
        algorithms: List,
        test_files: List[str],
    ) -> Dict[str, EfficiencyReport]:
        """
        对多个算法进行效率对比分析

        Args:
            algorithms: 降噪算法实例列表
            test_files: 测试音频文件列表

        Returns:
            {algorithm_name: EfficiencyReport}
        """
        results = {}

        for denoiser in algorithms:
            name = denoiser.name if hasattr(denoiser, 'name') else str(type(denoiser).__name__)
            logger.info(f"分析算法效率: {name}")
            try:
                report = self.profile_batch(denoiser, test_files)
                results[name] = report
            except Exception as e:
                logger.error(f"效率分析失败 {name}: {e}")

        return results

    # ===========================
    # 辅助方法
    # ===========================

    @staticmethod
    def _count_parameters(denoiser) -> int:
        """统计模型参数量"""
        try:
            model = getattr(denoiser, '_model', None) or getattr(denoiser, 'model', None)
            if model is None:
                return 0

            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
            elif hasattr(model, 'num_parameters'):
                return model.num_parameters()
            elif isinstance(model, dict):
                return sum(
                    p.numel() if hasattr(p, 'numel') else 0
                    for p in model.values()
                )
        except Exception:
            pass
        return 0

    @staticmethod
    def _estimate_model_size(denoiser) -> float:
        """估算模型文件大小 (MB)"""
        try:
            model = getattr(denoiser, '_model', None) or getattr(denoiser, 'model', None)
            if model is None:
                return 0.0

            if hasattr(model, 'state_dict'):
                import tempfile
                import torch
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                    torch.save(model.state_dict(), f.name)
                    size = os.path.getsize(f.name) / (1024 * 1024)
                os.unlink(f.name)
                return round(size, 2)
        except Exception:
            pass

        # 尝试从模型路径估算
        for attr in ['model_path', '_model_path', 'checkpoint']:
            path = getattr(denoiser, attr, None)
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    return round(os.path.getsize(path) / (1024 * 1024), 2)
                elif os.path.isdir(path):
                    total = 0
                    for root, _, files in os.walk(path):
                        for f in files:
                            try:
                                total += os.path.getsize(os.path.join(root, f))
                            except Exception:
                                pass
                    return round(total / (1024 * 1024), 2)

        return 0.0


# ===========================
# 便捷函数
# ===========================


def quick_efficiency_check(
    denoiser, audio_file: str, device_type: str = "auto"
) -> str:
    """
    快速效率检查，返回人类可读的摘要

    Args:
        denoiser: 降噪算法实例
        audio_file: 测试音频文件
        device_type: 设备类型

    Returns:
        可读的效率摘要
    """
    profiler = EfficiencyProfiler()
    metrics = profiler.profile_denoiser(denoiser, audio_file)

    lines = [
        f"=== 效率分析: {denoiser.name if hasattr(denoiser, 'name') else 'Unknown'} ===",
        f"处理时间:   {metrics.processing_time:.3f}s",
        f"音频时长:   {metrics.audio_duration:.2f}s",
        f"RTF:        {metrics.rtf:.4f} {'✓ 实时' if metrics.rtf < 1.0 else '✗ 非实时'}",
        f"峰值内存:   {metrics.peak_memory_mb:.1f} MB",
    ]

    if metrics.peak_gpu_memory_mb > 0:
        lines.append(f"GPU显存:    {metrics.peak_gpu_memory_mb:.1f} MB")
        lines.append(f"GPU利用率:  {metrics.gpu_utilization_pct:.1f}%")

    if metrics.model_params_count > 0:
        if metrics.model_params_count > 1e6:
            params_str = f"{metrics.model_params_count/1e6:.1f}M"
        else:
            params_str = f"{metrics.model_params_count/1e3:.1f}K"
        lines.append(f"模型参数:   {params_str}")

    lines.append(f"模型大小:   {metrics.model_size_mb:.1f} MB")

    return "\n".join(lines)
