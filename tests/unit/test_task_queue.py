"""
任务队列单元测试
测试任务队列管理功能
"""
import pytest
import asyncio
import os
from datetime import datetime


class TestTaskStatus:
    """任务状态测试类"""

    def test_task_status_enum(self):
        """测试任务状态枚举"""
        try:
            from app.core.task_queue import TaskStatus
            
            # 验证状态值
            assert TaskStatus.PENDING.value == "pending"
            assert TaskStatus.QUEUED.value == "queued"
            assert TaskStatus.PROCESSING.value == "processing"
            assert TaskStatus.COMPLETED.value == "completed"
            assert TaskStatus.FAILED.value == "failed"
        except ImportError as e:
            pytest.skip(f"任务队列模块导入失败: {e}")


class TestTask:
    """任务对象测试类"""

    def test_task_creation(self):
        """测试任务创建"""
        try:
            from app.core.task_queue import Task, TaskStatus
            
            task = Task(
                task_id="test_task_001",
                user="test_user",
                status=TaskStatus.PENDING
            )
            
            assert task.task_id == "test_task_001"
            assert task.user == "test_user"
            assert task.status == TaskStatus.PENDING
            assert task.progress == 0
            assert isinstance(task.created_at, datetime)
        except ImportError as e:
            pytest.skip(f"任务队列模块导入失败: {e}")

    def test_task_default_values(self):
        """测试任务默认值"""
        try:
            from app.core.task_queue import Task
            
            task = Task(task_id="test_task", user="user")
            
            # 验证默认值
            assert task.progress == 0
            assert task.message == ""
            assert task.result_file is None
            assert isinstance(task.data, dict)
        except ImportError as e:
            pytest.skip(f"任务队列模块导入失败: {e}")


class TestTaskQueue:
    """任务队列测试类"""

    @pytest.fixture
    def queue(self):
        """创建任务队列实例"""
        try:
            from app.core.task_queue import TaskQueue
            return TaskQueue(max_workers=1)
        except ImportError:
            pytest.skip("任务队列模块导入失败")

    @pytest.mark.asyncio
    async def test_queue_initialization(self, queue):
        """测试队列初始化"""
        assert queue.max_workers == 1
        assert queue.get_queue_size() == 0
        assert queue.get_processing_count() == 0

    @pytest.mark.asyncio
    async def test_submit_task(self, queue):
        """测试提交任务"""
        try:
            from app.core.task_queue import Task, TaskStatus
            
            task = Task(task_id="test_001", user="user1")
            result = await queue.submit(task)
            
            assert result is True
            assert queue.get_queue_size() == 1
            assert task.status == TaskStatus.QUEUED
        except ImportError:
            pytest.skip("任务队列模块导入失败")

    @pytest.mark.asyncio
    async def test_submit_duplicate_task(self, queue):
        """测试提交重复任务"""
        try:
            from app.core.task_queue import Task
            
            task = Task(task_id="test_dup", user="user1")
            result1 = await queue.submit(task)
            result2 = await queue.submit(task)
            
            assert result1 is True
            assert result2 is False  # 重复任务应该失败
        except ImportError:
            pytest.skip("任务队列模块导入失败")

    @pytest.mark.asyncio
    async def test_get_task(self, queue):
        """测试获取任务"""
        try:
            from app.core.task_queue import Task
            
            task = Task(task_id="test_get", user="user1")
            await queue.submit(task)
            
            retrieved = await queue.get_task("test_get")
            assert retrieved is not None
            assert retrieved.task_id == "test_get"
            
            # 获取不存在的任务
            not_found = await queue.get_task("nonexistent")
            assert not_found is None
        except ImportError:
            pytest.skip("任务队列模块导入失败")

    @pytest.mark.asyncio
    async def test_get_user_tasks(self, queue):
        """测试获取用户任务列表"""
        try:
            from app.core.task_queue import Task
            
            task1 = Task(task_id="test_user1_1", user="user1")
            task2 = Task(task_id="test_user1_2", user="user1")
            task3 = Task(task_id="test_user2_1", user="user2")
            
            await queue.submit(task1)
            await queue.submit(task2)
            await queue.submit(task3)
            
            user1_tasks = await queue.get_user_tasks("user1")
            assert len(user1_tasks) == 2
            
            user2_tasks = await queue.get_user_tasks("user2")
            assert len(user2_tasks) == 1
        except ImportError:
            pytest.skip("任务队列模块导入失败")

    @pytest.mark.asyncio
    async def test_update_task(self, queue):
        """测试更新任务"""
        try:
            from app.core.task_queue import Task
            
            task = Task(task_id="test_update", user="user1")
            await queue.submit(task)
            
            await queue.update_task("test_update", progress=50, message="处理中")
            
            updated = await queue.get_task("test_update")
            assert updated.progress == 50
            assert updated.message == "处理中"
        except ImportError:
            pytest.skip("任务队列模块导入失败")


class TestGlobalTaskQueue:
    """全局任务队列测试类"""

    def test_global_queue_exists(self):
        """测试全局队列实例存在"""
        try:
            from app.core.task_queue import task_queue, TaskQueue
            
            assert task_queue is not None
            assert isinstance(task_queue, TaskQueue)
        except ImportError as e:
            pytest.skip(f"任务队列模块导入失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
