"""
MOS计算性能优化模块
提供并行计算、缓存、批处理等优化策略
"""
import os
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import torch
import torch.cuda as cuda
from datetime import datetime

# 性能监控装饰器
def timing_decorator(func_name: str):
    """计算函数执行时间的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            return result, elapsed
        return wrapper
    return decorator


class PerformanceProfiler:
    """性能分析器 - 记录各阶段耗时"""
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start(self, stage: str):
        """开始计时某个阶段"""
        self.start_times[stage] = time.time()
    
    def end(self, stage: str):
        """结束计时某个阶段"""
        if stage in self.start_times:
            elapsed = time.time() - self.start_times[stage]
            if stage not in self.timings:
                self.timings[stage] = []
            self.timings[stage].append(elapsed)
            del self.start_times[stage]
            return elapsed
        return 0.0
    
    def get_report(self) -> Dict[str, Dict[str, float]]:
        """生成性能报告"""
        report = {}
        for stage, times in self.timings.items():
            report[stage] = {
                "total": sum(times),
                "avg": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "count": len(times)
            }
        return report
    
    def reset(self):
        """重置计时器"""
        self.timings.clear()
        self.start_times.clear()


class MOSComputeOptimizer:
    """MOS计算优化器 - 提供并行和批处理能力"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, os.cpu_count() or 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.profiler = PerformanceProfiler()
        
        # 模型缓存
        self._model_cache: Dict[str, any] = {}
        self._cache_lock = threading.Lock()
        
    def get_cached_model(self, model_name: str, factory: Callable) -> any:
        """获取缓存的模型实例"""
        with self._cache_lock:
            if model_name not in self._model_cache:
                self._model_cache[model_name] = factory()
            return self._model_cache[model_name]
    
    def clear_model_cache(self):
        """清空模型缓存"""
        with self._cache_lock:
            # 清理CUDA缓存
            if cuda.is_available():
                cuda.empty_cache()
            self._model_cache.clear()
    
    async def parallel_compute(
        self,
        compute_funcs: List[Tuple[Callable, tuple, dict]],
        use_processes: bool = False
    ) -> List[any]:
        """
        并行执行多个计算函数
        
        Args:
            compute_funcs: [(func, args, kwargs), ...]
            use_processes: 是否使用进程池(适用于CPU密集型)
        
        Returns:
            计算结果列表
        """
        loop = asyncio.get_event_loop()
        
        if use_processes:
            # 对于CPU密集型任务使用进程池
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    loop.run_in_executor(executor, func, *args, **kwargs)
                    for func, args, kwargs in compute_funcs
                ]
        else:
            # 对于IO密集型或GPU任务使用线程池
            futures = [
                loop.run_in_executor(self.executor, func, *args, **kwargs)
                for func, args, kwargs in compute_funcs
            ]
        
        return await asyncio.gather(*futures)
    
    def batch_process_files(
        self,
        files: List[str],
        process_func: Callable,
        batch_size: int = 4
    ) -> List[any]:
        """
        批量处理文件
        
        Args:
            files: 文件列表
            process_func: 处理函数
            batch_size: 批处理大小
        
        Returns:
            处理结果列表
        """
        results = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_results = [process_func(f) for f in batch]
            results.extend(batch_results)
        return results


class AudioPreprocessingOptimizer:
    """音频预处理优化器"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._cache_lock = threading.Lock()
        self.max_cache_size = 50  # 最多缓存50个音频文件
    
    def cache_audio(self, file_path: str, audio: np.ndarray, sr: int):
        """缓存加载的音频"""
        with self._cache_lock:
            if len(self._audio_cache) >= self.max_cache_size:
                # LRU: 移除最早的缓存
                oldest = next(iter(self._audio_cache))
                del self._audio_cache[oldest]
            self._audio_cache[file_path] = (audio, sr)
    
    def get_cached_audio(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """获取缓存的音频"""
        with self._cache_lock:
            return self._audio_cache.get(file_path)
    
    def clear_audio_cache(self):
        """清空音频缓存"""
        with self._cache_lock:
            self._audio_cache.clear()


# 全局优化器实例
mos_optimizer = MOSComputeOptimizer()
audio_optimizer = AudioPreprocessingOptimizer()


def get_performance_report() -> Dict:
    """获取性能报告"""
    return {
        "mos_compute": mos_optimizer.profiler.get_report(),
        "audio_preprocessing": audio_optimizer.profiler.get_report(),
        "timestamp": datetime.now().isoformat()
    }


def reset_performance_tracking():
    """重置性能跟踪"""
    mos_optimizer.profiler.reset()
    audio_optimizer.profiler.reset()
