"""
WebSocket 连接管理器（共享模块）

提供通用的 WebSocket 连接管理，供各 API 模块（mos、restoration 等）使用。
每个模块持自己的 ConnectionManager 实例，task_id 全局唯一 UUID，互不冲突。
"""

import logging
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("audiomos")


class ConnectionManager:
    """
    WebSocket 连接管理器

    管理 task_id → WebSocket 的映射，支持连接/断开/推送进度。
    每个业务模块（mos、restoration）应持独立实例。
    """

    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        """接受 WebSocket 连接并注册到 task_id"""
        await websocket.accept()
        self.active_connections[task_id] = websocket
        logger.debug(f"[WebSocket] 连接建立: task_id={task_id}, 当前连接数={len(self.active_connections)}")

    def disconnect(self, task_id: str):
        """断开 task_id 对应的连接"""
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            logger.debug(f"[WebSocket] 连接断开: task_id={task_id}, 当前连接数={len(self.active_connections)}")

    async def send_progress(self, task_id: str, data: dict):
        """向 task_id 对应的 WebSocket 推送进度数据"""
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(data)
            except Exception:
                # 连接已关闭，移除连接
                self.disconnect(task_id)
