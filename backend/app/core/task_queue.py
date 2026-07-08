"""
任务队列管理器
支持多用户并发提交任务，后台顺序执行
任务状态持久化到JSON文件，支持超时、自动重试、任务取消
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable

logger = logging.getLogger("audiomos")


class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """任务数据类"""
    task_id: str
    user: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result_file: Optional[str] = None
    data: Dict = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 0      # 0 = 不重试
    timeout: int = 0           # 0 = 无超时限制
    _cancelled: bool = False   # 内部标志，不入JSON

    def to_dict(self) -> dict:
        """序列化为JSON可序列化dict"""
        return {
            "task_id": self.task_id,
            "user": self.user,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result_file": self.result_file,
            "data": self.data,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        """从dict反序列化"""
        return cls(
            task_id=d["task_id"],
            user=d["user"],
            status=TaskStatus(d["status"]),
            progress=d.get("progress", 0),
            message=d.get("message", ""),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            result_file=d.get("result_file"),
            data=d.get("data", {}),
            retry_count=d.get("retry_count", 0),
            max_retries=d.get("max_retries", 0),
            timeout=d.get("timeout", 0),
        )


class TaskQueue:
    """任务队列管理器 — 支持JSON持久化、超时、自动重试、取消"""

    def __init__(self, max_workers: int = 1, persistence_dir: Optional[str] = None):
        self.max_workers = max_workers
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}        # 单数据源：所有任务
        self.processing: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._processor: Optional[Callable] = None
        self._running = False
        self._persistence_dir = Path(persistence_dir) if persistence_dir else None
        if self._persistence_dir:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)

    # ── 持久化 ──────────────────────────────────────────

    async def _persist_task(self, task: Task):
        """将单个任务写入JSON文件（调用方需持有锁）"""
        if not self._persistence_dir:
            return
        task_file = self._persistence_dir / f"{task.task_id}.json"
        try:
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[TaskQueue] 持久化失败 {task.task_id}: {e}")

    async def _remove_persisted_task(self, task_id: str):
        """删除任务对应的JSON文件"""
        if not self._persistence_dir:
            return
        task_file = self._persistence_dir / f"{task_id}.json"
        try:
            if task_file.exists():
                task_file.unlink()
        except Exception as e:
            logger.error(f"[TaskQueue] 删除持久化文件失败 {task_id}: {e}")

    async def load_persisted(self):
        """启动时从JSON目录恢复未完成任务"""
        if not self._persistence_dir or not self._persistence_dir.exists():
            return
        loaded = 0
        for task_file in sorted(self._persistence_dir.glob("*.json")):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                task = Task.from_dict(d)
                # 只恢复非终态任务（重启前在排队或处理中的）
                if task.status in (TaskStatus.PROCESSING, TaskStatus.QUEUED,
                                   TaskStatus.PENDING):
                    task.status = TaskStatus.QUEUED
                    task.message = "已从持久化恢复，等待处理..."
                    task.progress = 0
                    task.retry_count = 0  # 重置重试计数
                    self.tasks[task.task_id] = task
                    await self.queue.put(task)
                    loaded += 1
                    logger.info(f"[TaskQueue] 恢复任务: {task.task_id}")
            except Exception as e:
                logger.error(f"[TaskQueue] 加载任务文件失败 {task_file}: {e}")
        if loaded:
            logger.info(f"[TaskQueue] 从持久化恢复 {loaded} 个任务")

    # ── 生命周期 ────────────────────────────────────────

    async def start(self, processor: Callable):
        """启动队列处理器"""
        self._processor = processor
        self._running = True
        await self.load_persisted()  # 先恢复历史任务，再启动worker
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

    # ── Worker ──────────────────────────────────────────

    async def _worker(self):
        """后台工作协程：取任务 → 执行（含超时/重试）"""
        while self._running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # 任务被取消或已删除，跳过
            async with self._lock:
                if task._cancelled or task.task_id not in self.tasks:
                    continue
                task.status = TaskStatus.PROCESSING
                task.message = "正在处理..."
                task.updated_at = datetime.now()
                self.processing[task.task_id] = task
                await self._persist_task(task)

            logger.info(f"[TaskQueue] 开始处理: {task.task_id}")

            # ── 执行（带可选超时） ──
            try:
                if self._processor:
                    if task.timeout > 0:
                        await asyncio.wait_for(
                            self._processor(task), timeout=task.timeout
                        )
                    else:
                        await self._processor(task)
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                task.message = "处理完成"

            except asyncio.TimeoutError:
                logger.error(f"[TaskQueue] 超时 {task.task_id} ({task.timeout}s)")
                task.status = TaskStatus.TIMEOUT
                task.message = f"处理超时 ({task.timeout}s)"

            except Exception as e:
                logger.error(f"[TaskQueue] 失败 {task.task_id}: {e}")
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.QUEUED
                    task.message = f"重试 {task.retry_count}/{task.max_retries}"
                    task.progress = 0
                    logger.info(f"[TaskQueue] {task.task_id} 将重试 "
                                f"({task.retry_count}/{task.max_retries})")
                    await self.queue.put(task)
                else:
                    task.status = TaskStatus.FAILED
                    reason = str(e)[:200]
                    if task.retry_count > 0:
                        task.message = f"处理失败 (已重试{task.retry_count}次): {reason}"
                    else:
                        task.message = f"处理失败: {reason}"

            task.updated_at = datetime.now()

            async with self._lock:
                self.tasks[task.task_id] = task
                if task.task_id in self.processing:
                    del self.processing[task.task_id]
                await self._persist_task(task)

            logger.info(f"[TaskQueue] 完成: {task.task_id}, 状态: {task.status.value}")

    # ── 任务提交与操作 ──────────────────────────────────

    async def submit(self, task: Task) -> bool:
        """提交任务到队列"""
        async with self._lock:
            if task.task_id in self.tasks:
                logger.warning(f"[TaskQueue] 任务已存在: {task.task_id}")
                return False
            task.status = TaskStatus.QUEUED
            task.message = "等待处理..."
            task.updated_at = datetime.now()
            self.tasks[task.task_id] = task
            await self._persist_task(task)
        await self.queue.put(task)
        logger.info(f"[TaskQueue] 提交: {task.task_id}, 用户: {task.user}")
        return True

    async def cancel(self, task_id: str) -> bool:
        """取消任务（排队中/处理中均可取消）"""
        async with self._lock:
            if task_id not in self.tasks:
                return False
            task = self.tasks[task_id]
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                               TaskStatus.TIMEOUT, TaskStatus.CANCELLED):
                return False
            task._cancelled = True
            task.status = TaskStatus.CANCELLED
            task.message = "已取消"
            task.updated_at = datetime.now()
            self.tasks[task_id] = task
            await self._persist_task(task)
            logger.info(f"[TaskQueue] 已取消: {task_id}")
            return True

    async def delete_task(self, task_id: str):
        """删除任务（从内存和持久化中移除）"""
        async with self._lock:
            self.tasks.pop(task_id, None)
            self.processing.pop(task_id, None)
        await self._remove_persisted_task(task_id)

    # ── 查询 ────────────────────────────────────────────

    async def get_task(self, task_id: str) -> Optional[Task]:
        async with self._lock:
            return self.tasks.get(task_id)

    async def get_user_tasks(self, user: str) -> List[Task]:
        async with self._lock:
            return [t for t in self.tasks.values() if t.user == user]

    async def get_all_tasks(self) -> List[Task]:
        async with self._lock:
            return list(self.tasks.values())

    async def update_task(self, task_id: str, **kwargs):
        """更新任务字段并持久化"""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                task.updated_at = datetime.now()
                await self._persist_task(task)

    def get_queue_size(self) -> int:
        return self.queue.qsize()

    def get_processing_count(self) -> int:
        return len(self.processing)


# 全局任务队列实例（MOS评分使用）
# max_workers 从 config.yaml task_queue.max_workers 读取，默认 1
# 注意：调高此值需要确保各评分器支持并发调用（模型实例线程安全）
_task_queue_max_workers = 1
try:
    from app.core.config import settings
    _task_queue_max_workers = getattr(settings.task_queue, 'max_workers', 1)
except Exception:
    pass
task_queue = TaskQueue(max_workers=_task_queue_max_workers)
