#!/usr/bin/env python3
"""
AudioMOS 后端服务启动脚本
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from app.core.config import settings
from app.core.logging_config import logger


def main():
    """
    主函数 - 启动服务
    """
    logger.info(f"启动 AudioMOS 后端服务...")
    logger.info(f"监听地址: {settings.server.backend.host}:{settings.server.backend.port}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.server.backend.host,
        port=settings.server.backend.port,
        reload=settings.server.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
