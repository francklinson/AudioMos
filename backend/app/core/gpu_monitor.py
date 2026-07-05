"""
GPU显存监控守护线程
长期运行防护,防止显存泄漏
"""
import time
import threading
import logging
from typing import Optional, Callable, Dict, List, Any

logger = logging.getLogger("audiomos")


class GPUMemoryMonitor:
    """GPU显存监控守护线程"""
    
    def __init__(
        self,
        check_interval: float = 60.0,
        warning_threshold_mb: float = 20_000,
        critical_threshold_mb: float = 23_000,
        cleanup_callback: Optional[Callable] = None,
    ):
        """
        Args:
            check_interval: 检查间隔(秒)
            warning_threshold_mb: 警告阈值(MB)
            critical_threshold_mb: 严重阈值(MB)
            cleanup_callback: 清理回调函数
        """
        self.check_interval = check_interval
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.cleanup_callback = cleanup_callback
        self._running = False
        self._thread = None
        self._history: List[Dict[str, Any]] = []  # 显存历史记录
        self._cuda_available = False
        
        # 检查CUDA可用性
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                logger.info(f"[GPU监控] CUDA可用 - 设备: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("[GPU监控] CUDA不可用,监控线程将跳过启动")
        except ImportError:
            logger.warning("[GPU监控] torch未安装,监控线程将跳过启动")
            self._cuda_available = False
        
    def start(self):
        """启动监控线程"""
        if not self._cuda_available:
            logger.warning("[GPU监控] CUDA不可用,监控线程跳过启动")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"[GPU监控] 监控线程已启动 "
            f"(间隔={self.check_interval}s, "
            f"警告阈值={self.warning_threshold_mb:.0f}MB, "
            f"严重阈值={self.critical_threshold_mb:.0f}MB)"
        )
        
    def stop(self):
        """停止监控线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[GPU监控] 监控线程已停止")
        
    def _monitor_loop(self):
        """监控循环(后台线程)"""
        import torch
        
        while self._running:
            try:
                # 获取显存状态
                allocated_mb = torch.cuda.memory_allocated() / 1024**2
                reserved_mb = torch.cuda.memory_reserved() / 1024**2
                total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                
                # 记录历史
                self._history.append({
                    "timestamp": time.time(),
                    "allocated_mb": allocated_mb,
                    "reserved_mb": reserved_mb,
                    "utilization_pct": (allocated_mb / total_mb) * 100,
                })
                
                # 保留最近1小时数据(最多60条)
                if len(self._history) > 60:
                    self._history.pop(0)
                
                # ── 状态判断 ──
                if allocated_mb > self.critical_threshold_mb:
                    logger.critical(
                        f"[GPU监控] ⚠️ 显存严重超限: {allocated_mb:.1f}MB "
                        f"(阈值={self.critical_threshold_mb:.1f}MB, "
                        f"利用率={allocated_mb/total_mb*100:.1f}%)"
                    )
                    
                    # 触发紧急清理
                    if self.cleanup_callback:
                        logger.critical("[GPU监控] 执行紧急清理回调...")
                        try:
                            self.cleanup_callback()
                        except Exception as e:
                            logger.error(f"[GPU监控] 清理回调执行失败: {e}")
                    
                    # 强制清理缓存池
                    torch.cuda.empty_cache()
                    logger.critical("[GPU监控] 已强制清理缓存池碎片")
                    
                    # 再次检查
                    allocated_after = torch.cuda.memory_allocated() / 1024**2
                    freed_mb = allocated_mb - allocated_after
                    logger.critical(
                        f"[GPU监控] 清理后显存: {allocated_after:.1f}MB "
                        f"(释放碎片: {freed_mb:.1f}MB)"
                    )
                    
                elif allocated_mb > self.warning_threshold_mb:
                    logger.warning(
                        f"[GPU监控] ⚠️ 显存占用偏高: {allocated_mb:.1f}MB "
                        f"(阈值={self.warning_threshold_mb:.1f}MB, "
                        f"利用率={allocated_mb/total_mb*100:.1f}%)"
                    )
                    
                    # 定期生成报告(每5次检查)
                    if len(self._history) % 5 == 0:
                        self._generate_report()
                
                # ── 检测显存增长趋势 ──
                if len(self._history) >= 10:
                    recent = self._history[-10:]
                    time_span = recent[-1]["timestamp"] - recent[0]["timestamp"]
                    growth_rate = (
                        (recent[-1]["allocated_mb"] - recent[0]["allocated_mb"]) 
                        / time_span * 60  # MB/min
                    )
                    
                    if growth_rate > 100:  # 每分钟增长>100MB
                        logger.warning(
                            f"[GPU监控] 检测到显存持续增长: {growth_rate:.1f}MB/min"
                        )
                        logger.warning("[GPU监控] 可能存在显存泄漏,建议重启服务")
                        
                        # 触发预防性清理
                        if self.cleanup_callback:
                            logger.warning("[GPU监控] 执行预防性清理...")
                            self.cleanup_callback()
                
            except Exception as e:
                logger.error(f"[GPU监控] 监控异常: {e}")
                import traceback
                logger.error(f"[GPU监控] 错误详情: {traceback.format_exc()}")
                
            time.sleep(self.check_interval)
    
    def _generate_report(self):
        """生成显存趋势报告"""
        if len(self._history) < 5:
            return
        
        recent = self._history[-5:]
        
        logger.info("[GPU监控] 显存趋势报告(最近5分钟):")
        for i, record in enumerate(recent, 1):
            logger.info(
                f"  {i}. {record['allocated_mb']:.1f}MB "
                f"/ {record['reserved_mb']:.1f}MB "
                f"(利用率: {record['utilization_pct']:.1f}%)"
            )
        
        # 计算变化趋势
        change = recent[-1]["allocated_mb"] - recent[0]["allocated_mb"]
        trend = "稳定" if abs(change) < 50 else ("增长" if change > 0 else "下降")
        
        logger.info(f"[GPU监控] 趋势: {trend} (变化{change:.1f}MB)")
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前显存状态"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {
                    "cuda_available": False,
                    "message": "CUDA不可用"
                }
            
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            
            return {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "total_mb": round(total_mb, 1),
                "allocated_mb": round(allocated_mb, 1),
                "reserved_mb": round(reserved_mb, 1),
                "utilization_pct": round((allocated_mb / total_mb) * 100, 1),
                "history_length": len(self._history),
                "thresholds": {
                    "warning_mb": self.warning_threshold_mb,
                    "critical_mb": self.critical_threshold_mb,
                },
                "status": (
                    "critical" if allocated_mb > self.critical_threshold_mb
                    else "warning" if allocated_mb > self.warning_threshold_mb
                    else "normal"
                ),
            }
        except Exception as e:
            logger.error(f"[GPU监控] 获取状态失败: {e}")
            return {
                "cuda_available": False,
                "error": str(e)
            }
    
    def get_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """获取最近N次显存记录"""
        return self._history[-last_n:] if len(self._history) >= last_n else self._history


# ── 紧急清理回调函数 ──
def emergency_gpu_cleanup():
    """紧急显存清理(当严重超限时执行)"""
    import torch
    
    logger.critical("[GPU紧急清理] 开始清理...")
    
    # 1. 清理缓存池碎片
    torch.cuda.empty_cache()
    
    # 2. 同步GPU操作(确保所有操作完成)
    torch.cuda.synchronize()
    
    # 3. 再次清理
    torch.cuda.empty_cache()
    
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    logger.critical(f"[GPU紧急清理] 清理完成 - 当前显存: {allocated_mb:.1f}MB")
    
    # 4. 建议: 如果显存仍然紧张,可以考虑清理长期未用的模型实例
    # TODO: 实现LRU清理逻辑(需要RestorationRegistry配合)
    # from app.api.restoration import _restorer_instances
    # 根据使用频率清理模型...


# ── 全局实例 ──
gpu_monitor: Optional[GPUMemoryMonitor] = None


def init_gpu_monitor():
    """初始化GPU监控实例"""
    global gpu_monitor
    
    if gpu_monitor is not None:
        logger.warning("[GPU监控] 监控实例已存在,跳过重复初始化")
        return gpu_monitor
    
    gpu_monitor = GPUMemoryMonitor(
        check_interval=60.0,       # 每分钟检查一次
        warning_threshold_mb=20_000,  # 20GB警告
        critical_threshold_mb=23_000, # 23GB严重(24GB显存的95%)
        cleanup_callback=emergency_gpu_cleanup,
    )
    
    return gpu_monitor


def get_gpu_monitor() -> Optional[GPUMemoryMonitor]:
    """获取GPU监控实例"""
    return gpu_monitor