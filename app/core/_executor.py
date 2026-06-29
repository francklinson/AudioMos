"""
全局共享线程池模块
统一管理项目中所有并行任务的线程池，避免创建多个独立线程池导致资源争用。

用法:
    from app.core._executor import get_shared_executor
    executor = get_shared_executor(max_workers=8)
    future = executor.submit(fn, arg)
"""
import threading
from concurrent.futures import ThreadPoolExecutor

_SHARED_EXECUTOR = None
_LOCK = threading.Lock()


def get_shared_executor(max_workers: int = 8) -> ThreadPoolExecutor:
    """
    获取全局共享线程池（延迟初始化，应用生命周期内保持）

    Args:
        max_workers: 最大线程数。默认为8，覆盖旧版各模块的独立4路配置。
            增大后可同时容纳MOS评分(4路) + 匹配切分(4路)的并发需求。

    Returns:
        全局共享ThreadPoolExecutor实例
    """
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None:
        with _LOCK:
            if _SHARED_EXECUTOR is None:
                _SHARED_EXECUTOR = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix='audiomos_shared'
                )
    return _SHARED_EXECUTOR
