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
from app.core.network_utils import validate_and_fix_host, print_network_info


def main():
    """
    主函数 - 启动服务
    """
    logger.info("=" * 60)
    logger.info("启动 AudioMOS 后端服务...")
    logger.info("=" * 60)
    
    # 获取配置的 host
    configured_host = settings.server.host
    configured_port = settings.server.port
    
    # 如果配置的是 auto，进行自动检测
    if configured_host.lower() == "auto":
        actual_host, warning = validate_and_fix_host(configured_host, "后端服务")
        if warning:
            logger.warning(warning)
        logger.info(f"监听地址: {actual_host}:{configured_port}")
    else:
        # 直接使用配置的值 (像 VersTTS 一样)
        actual_host = configured_host
        logger.info(f"监听地址: {actual_host}:{configured_port}")
    
    # 启动服务
    uvicorn.run(
        "app.main:app",
        host=actual_host,
        port=configured_port,
        reload=settings.server.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
