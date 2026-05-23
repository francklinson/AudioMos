"""
任务队列管理器
支持多用户并发提交任务，后台顺序执行
"""
import asyncio
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

logger = logging.getLogger("audiomos")


class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    user: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result_file: Optional[str] = None
    data: Dict = field(default_factory=dict)


class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}
        self.processing: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._processor: Optional[Callable] = None
        self._running = False
        
    async def start(self, processor: Callable):
        """启动队列处理器"""
        self._processor = processor
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("任务队列已启动")
        
    async def stop(self):
        """停止队列处理器"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("任务队列已停止")
        
    async def _worker(self):
        """后台工作线程"""
        while self._running:
            try:
                # 获取队列中的任务
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                async with self._lock:
                    if task.task_id in self.tasks:
                        task.status = TaskStatus.PROCESSING
                        task.message = "正在处理..."
                        task.updated_at = datetime.now()
                        self.processing[task.task_id] = task
                        
                logger.info(f"开始处理任务: {task.task_id}")
                
                # 执行处理
                try:
                    if self._processor:
                        await self._processor(task)
                        task.status = TaskStatus.COMPLETED
                        task.progress = 100
                        task.message = "处理完成"
                except Exception as e:
                    logger.error(f"任务处理失败 {task.task_id}: {e}")
                    task.status = TaskStatus.FAILED
                    task.message = f"处理失败: {str(e)}"
                    
                task.updated_at = datetime.now()
                
                async with self._lock:
                    self.tasks[task.task_id] = task
                    if task.task_id in self.processing:
                        del self.processing[task.task_id]
                        
                logger.info(f"任务处理完成: {task.task_id}, 状态: {task.status.value}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"工作线程错误: {e}")
                
    async def submit(self, task: Task) -> bool:
        """提交任务到队列"""
        async with self._lock:
            if task.task_id in self.tasks:
                logger.warning(f"任务已存在: {task.task_id}")
                return False
                
            task.status = TaskStatus.QUEUED
            task.message = "等待处理..."
            task.updated_at = datetime.now()
            self.tasks[task.task_id] = task
            
        await self.queue.put(task)
        logger.info(f"任务已提交到队列: {task.task_id}, 用户: {task.user}")
        return True
        
    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务状态"""
        async with self._lock:
            return self.tasks.get(task_id)
            
    async def get_user_tasks(self, user: str) -> List[Task]:
        """获取用户的所有任务"""
        async with self._lock:
            return [t for t in self.tasks.values() if t.user == user]
            
    async def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        async with self._lock:
            return list(self.tasks.values())
            
    async def update_task(self, task_id: str, **kwargs):
        """更新任务状态"""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                task.updated_at = datetime.now()
                
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()
        
    def get_processing_count(self) -> int:
        """获取正在处理的任务数"""
        return len(self.processing)


# 全局任务队列实例
task_queue = TaskQueue(max_workers=1)
